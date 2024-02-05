#!/usr/bin/env python3

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstores/db_faiss'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# create vector database
def createVectorDatabase():
  loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
  documents = loader.load()
  textSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  texts = textSplitter.split_documents(documents)
  embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
  db = FAISS.from_documents(texts, embeddings)
  db.save_local(DB_FAISS_PATH)

createVectorDatabase()