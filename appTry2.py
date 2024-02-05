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
    
# def init_qa_bot():
#     if 'qa_bot' not in st.session_state:
#         embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
#                                            model_kwargs={'device': 'cuda'})
#         db = FAISS.load_local(DB_FAISS_PATH, embeddings)
#         llm = loadLlm()
#         qaPrompt = setCustomPrompt()
#         qa = retrievalQaChain(llm, qaPrompt, db)
#         st.session_state.qa_bot = qa
#         st.session_state.qa_bot_initialized = True  # Mark as initialized

# if 'qa_bot_initialized' not in st.session_state:
#     init_qa_bot()

# Function to build 
def setCustomPrompt():
    customPromptTemplate = """use the following pieces of information to answer the user's question.
    If you don't know the answer, please say that you don't know the answer, don't try to make up
    an answer

    Context: {}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=customPromptTemplate, input_variables=['context', 'question'])
    return prompt


def loadLlm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv('OPENAI_API_KEY'))        

def retrievalQaChain(llm, prompt, db):
    qaChain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 4}),
        return_source_documents=True,
    )
    return qaChain



# Store LLM generated responses
# if "messages" not in st.session_state.keys():
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
#                                     model_kwargs={'device': 'cuda'})
#     db = FAISS.load_local(DB_FAISS_PATH, embeddings)
#     llm = loadLlm()
#     qaPrompt = setCustomPrompt()
#     qa = retrievalQaChain(llm, qaPrompt, db)
#     print("Passed here 1")
#     time.sleep(65)
#     print("Passed here 2")
#     print(qa.batch("what is Eczema?"))
#     # st.session_state.qa_bot = qa
#     # st.session_state.qa_bot_initialized = True  # Mark as initialized
#     st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# print("Session state is: ", st.session_state)

# Function to initialize and load the QA bot into session state
def init_qa_bot():
    if 'qa_bot_initialized' not in st.session_state:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cuda'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        llm = loadLlm()
        qaPrompt = setCustomPrompt()
        qa = retrievalQaChain(llm, qaPrompt, db)
        print(qa.apply("What is ECZMA?"))
        st.session_state.qa_bot = qa
        st.session_state.qa_bot_initialized = True  # Mark as initialized
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]
        
# Call the init function at the start
init_qa_bot()


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])



# Function for generating LLM response
def generate_response(prompt_input):
    # Use the QA bot from the session state
    qa = st.session_state.qa_bot
    response = qa.apply(prompt_input)
    # print("prompt_input", prompt_input)
    return response

# User-provided prompt
# if prompt := st.chat_input(disabled=not st.session_state.qa_bot_initialized):
if prompt := st.chat_input():
# prompt = st.chat_input()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    print("Message is:", message)
    st.session_state.messages.append(message)