import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import time


load_dotenv()

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstores/db_faiss'

# App title
st.set_page_config(page_title="🤗💬 HugChat")