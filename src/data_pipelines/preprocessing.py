import os
import argparse
import langchain
import wandb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

model_name = "jinaai/jina-embeddings-v2-small-en"

embeddings = HuggingFaceEmbeddings(
    model_name=model_name
)

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

with wandb.init(project="LLMOps-Pycon2024",name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:
    file_path = "src/data/1810.04805v2.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(len(docs))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings,persist_directory="src/rag_db")
    # 📦 save the vector database to the artifact
    vectorstore_artifact = wandb.Artifact(
        "vector-database", type="dataset", description="Vector Database for RAG model",
        metadata={"source": file_path,
                  "sizes": len(docs),
                  "embedding": model_name,
                  "chunks": len(splits),
                  "destined_for": "rag-model"})
    vectorstore_artifact.add_dir("src/rag_db")
    run.log_artifact(vectorstore_artifact)