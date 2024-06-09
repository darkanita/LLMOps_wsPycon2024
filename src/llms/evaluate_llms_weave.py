import json
import openai
import weave
from weave import Model
from openai import AsyncAzureOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
import wandb
import argparse
import os

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
embeddings_deployment = os.environ["EMBEDDINGS_DEPLOYMENT_NAME"]
chat_deployment = os.environ["CHAT_COMPLETIONS_DEPLOYMENT_NAME"]
key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = os.environ["OPENAI_API_VERSION"]

# Create the embedding function
embedding_function = AzureOpenAIEmbeddings(
    azure_deployment=embeddings_deployment,
    openai_api_version=api_version)

# Create llm
llm = AzureChatOpenAI(
    openai_api_version=api_version,
    azure_deployment=chat_deployment,
)

# Initialize Weave
weave.init("LLMOps-Pycon2024")

# Retrieve the dataset
dataset_ref = weave.ref('bert-paper-qna').get()

# Download Vector Database
run = wandb.init("LLMOps-Pycon2024")
artifact = run.use_artifact('labsirius/LLMOps-Pycon2024/vector-database:v3', type='dataset')
artifact_dir = artifact.download()
vectorstore = Chroma(persist_directory="artifacts/vector-database-v3", embedding_function=embedding_function)
print(vectorstore.similarity_search("What is the name of the new language representation model introduced in the document?"))

@weave.op()
def get_most_relevant_document(query):
    articles = vectorstore.similarity_search(query)
    return articles

class RAGModel(Model):
    model_name: str
    api_version: str

    @weave.op()
    def predict(self, question: str) -> dict: # note: `question` will be used later to select data from our evaluation rows
        context = get_most_relevant_document(question)
        llm = AzureChatOpenAI(
            openai_api_version=self.api_version,
            azure_deployment=self.model_name,
        )
        run = wandb.init("LLMOps-Pycon2024")
        artifact = run.use_artifact('labsirius/LLMOps-Pycon2024/prompt_template:v0', type='dataset')
        artifact_dir = artifact.download()
        
        with open(os.path.join(artifact_dir, "prompt_template.txt"), 'r') as file:
          prompt_template = file.read()

        prompt = PromptTemplate(
          template=prompt_template, input_variables=["context", "question"]
        )
        
        chain = prompt | llm

        response = chain.invoke({
            "question": question,
            "context": "\n".join([doc.page_content for doc in context])
        })
        print("Respuesta", response)
        return {'answer': response, 'context': context}

# Initialize Weave
weave.init("LLMOps_wsPycon2024-src_llms")
model = RAGModel(
    model_name=chat_deployment,
    api_version=api_version
)
model.predict("What is the name of the new language representation model introduced in the document?")