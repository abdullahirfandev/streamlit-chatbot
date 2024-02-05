import streamlit as st
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths for data and FAISS DB
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstores/db_faiss'

# Custom prompt template
customPromptTemplate = """use the following pieces of information to answer the user's question.
If you don't know the answer, please say that you don't know the answer, don't try to make up
an answer

Context: {}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Streamlit page config
st.set_page_config(page_title="🤗💬 HugChat")

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "How may I help you?"}]

# Function to set custom prompt
def setCustomPrompt():
    prompt = PromptTemplate(template=customPromptTemplate, input_variables=['context', 'question'])
    return prompt

# Function to load the OpenAI LLM
def loadLlm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv('OPENAI_API_KEY'))

# Function to create the retrieval QA chain
def retrievalQaChain(llm, prompt, db):
    qaChain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
    )
    return qaChain

# Initialize the QA bot once and store it in the session state
def init_qa_bot():
    if 'qa_bot' not in st.session_state:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        llm = loadLlm()
        qaPrompt = setCustomPrompt()
        qa = retrievalQaChain(llm, qaPrompt, db)
        st.session_state['qa_bot'] = qa

# Call init_qa_bot function to ensure QA bot is initialized
init_qa_bot()

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        st.write(f"{message['role']}: {message['content']}")

# Input from the user
prompt = st.text_input('Enter your question:', '')

# Function to generate response
def generate_response(prompt_input):
    qa = st.session_state['qa_bot']
    response = qa({'query': prompt_input})  # Assuming 'qa' function accepts a dict with 'query'
    return response

# Handle the user input
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": str(response)})  # Convert response to string if necessary

# Update the chat display based on new entries in session state
