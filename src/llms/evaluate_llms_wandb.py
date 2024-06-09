import weave
import json
import os
import argparse
import wandb

from weave import Dataset

from openai import AzureOpenAI
from openai import AsyncAzureOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma

from rich.markdown import Markdown

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


# Upload Prompt
prompt_template = """
By using the following pieces of context, Please answer the user's question.
Provide a detailed answer whenever possible.
If the question is not related to BERT or you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Begin
==========
Question: {question}
Answer:"""

file_name = "prompt_template.txt"
with open(file_name, 'w') as file:
    file.write(prompt_template)

with wandb.init(project="LLMOps-Pycon2024",name=f"Prompt-Template ExecId-{args.IdExecution}", job_type="prompting-engineering") as run:
    artifact = wandb.Artifact("prompt_template",type="dataset")
    artifact.add_file(local_path="prompt_template.txt", name="prompt_template.txt")
    run.log_artifact(artifact)

prompt_template = """
By using the following pieces of context, Please answer the user's question.
If the question is not related to BERT or you don't know the answer, just say that you don't know, don't try to make up an answer.

Here is the context:{context}
Question: {question}
Answer:"""

file_name = "prompt_template.txt"
with open(file_name, 'w') as file:
    file.write(prompt_template)

with wandb.init(project="LLMOps-Pycon2024",name=f"Prompt-Template ExecId-{args.IdExecution}", job_type="prompting-engineering") as run:
    artifact = wandb.Artifact("prompt_template",type="dataset")
    artifact.add_file(local_path="prompt_template.txt", name="prompt_template.txt")
    run.log_artifact(artifact)

# Questions to Evaluate
# Initialize Weave
weave.init("LLMOps-Pycon2024")
# Create a dataset
dataset = Dataset(name='bert-paper-qna', rows=[
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "What is the name of the new language representation model introduced in the document?", "answer": "BERT", "context": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "What is the main difference between BERT and previous language representation models?", "answer": "BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.", "context": "Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "What is the advantage of fine-tuning BERT over using feature-based approaches?", "answer": "Fine-tuning BERT reduces the need for many heavily-engineered taskspecific architectures and transfers all parameters to initialize end-task model parameters.", "context": "We show that pre-trained representations reduce the need for many heavily-engineered taskspecific architectures. BERT is the first finetuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "What are the two unsupervised tasks used to pre-train BERT?", "answer": "Masked LM and next sentence prediction", "context": "In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a \"masked LM\" (MLM), although it is often referred to as a Cloze task in the literature (Taylor, 1953). In addition to the masked language model, we also use a \"next sentence prediction\" task that jointly pretrains text-pair representations."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "How does BERT handle single sentence and sentence pair inputs?", "answer": "It uses a special classification token ([CLS]) at the beginning of every input sequence and a special separator token ([SEP]) to separate sentences or mark the end of a sequence.", "context": "To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., h Question, Answeri) in one token sequence. The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "What are the three types of embeddings used to construct the input representation for BERT?", "answer": "Token embeddings, segment embeddings and position embeddings", "context": "For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings. A visualization of this construction can be seen in Figure 2."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "What is the size of the vocabulary used by BERT?", "answer": "30,000", "context": "We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "What are the two model sizes reported in the paper for BERT?", "answer": "BERTBASE (L=12, H=768, A=12, Total Parameters=110M) and BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M)", "context": "We primarily report results on two model sizes: BERTBASE (L=12, H=768, A=12, Total Parameters=110M) and BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M)."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "How does BERT predict the start and end positions of an answer span in SQuAD?", "answer": "It uses two vectors S and E whose dot products with the final hidden vectors of each token denote scores for start and end positions.", "context": "We only introduce a start vector S ∈ R H and an end vector E ∈ R H during fine-tuning. The probability of word i being the start of the answer span is computed as a dot product between Ti and S followed by a softmax over all of the words in the paragraph: Pi = e S·Ti P j e S·Tj . The analogous formula is used for the end of the answer span."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "What is the main benefit of using a masked language model over a standard left-to-right or right-to-left language model?", "answer": "It enables the representation to fuse the left and the right context, which allows to pretrain a deep bidirectional Transformer.", "context": "Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pretrain a deep bidirectional Transformer."},
    {"pdf_url":"https://arxiv.org/pdf/1810.04805.pdf", "chat_history":[], "question": "How much does GPT4 API cost?", "answer": "I don't know"}
])

# Publish the dataset
weave.publish(dataset)

# Retrieve the dataset
dataset_ref = weave.ref('bert-paper-qna').get()

# Access a specific example
#example_label = dataset_ref.rows[2]['question']


# TEST1
config = {
  "model_name": "gpt-4", #gpt-4, gpt-3.5-turbo, text-davinci-003, ...
  "prompt_version":'labsirius/LLMOps-Pycon2024/prompt_template:v1'
}

with wandb.init(project="LLMOps-Pycon2024",name=f"Evaluate-RAG ExecId-{args.IdExecution}", job_type="evaluate-rag") as run:
  config = wandb.config
  
  artifact = run.use_artifact('labsirius/LLMOps-Pycon2024/vector-database:v3', type='dataset')
  artifact_dir = artifact.download()
  
  vectorstore = Chroma(persist_directory="artifacts/vector-database-v3", embedding_function=embedding_function)
  print(vectorstore.similarity_search("What is the name of the new language representation model introduced in the document?"))
  
  artifact = run.use_artifact('labsirius/LLMOps-Pycon2024/prompt_template:v0', type='dataset')
  artifact_dir = artifact.download()

  with open(os.path.join(artifact_dir, "prompt_template.txt"), 'r') as file:
    prompt_template = file.read()

  prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
  )

  chain = prompt | llm

  for row in dataset_ref.rows:
        context_docs = vectorstore.similarity_search(row['question'])
        result = chain.invoke({
            "question": row['question'],
            "context": "\n".join([doc.page_content for doc in context_docs])
        })
        print(result)

with wandb.init(project="LLMOps-Pycon2024",name=f"Evaluate-RAG ExecId-{args.IdExecution}", job_type="evaluate-rag") as run:
  config = wandb.config
  
  artifact = run.use_artifact('labsirius/LLMOps-Pycon2024/vector-database:v3', type='dataset')
  artifact_dir = artifact.download()
  
  vectorstore = Chroma(persist_directory="artifacts/vector-database-v3", embedding_function=embedding_function)
  print(vectorstore.similarity_search("What is the name of the new language representation model introduced in the document?"))
  
  artifact = run.use_artifact('labsirius/LLMOps-Pycon2024/prompt_template:v1', type='dataset')
  artifact_dir = artifact.download()

  with open(os.path.join(artifact_dir, "prompt_template.txt"), 'r') as file:
    prompt_template = file.read()

  prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
  )

  chain = prompt | llm

  for row in dataset_ref.rows:
        context_docs = vectorstore.similarity_search(row['question'])
        result = chain.invoke({
            "question": row['question'],
            "context": "\n".join([doc.page_content for doc in context_docs])
        })
        print(result)