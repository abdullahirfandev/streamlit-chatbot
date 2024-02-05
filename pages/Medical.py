from streaming import StreamHandler
import utils
import os
import streamlit as st
# from langchain_community.llms import openai
# from langchain.llms import openai
from langchain.chains import ConversationChain, RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS




st.set_page_config(page_title="Medical Chatbot", page_icon="💬")
st.header("Simple Chatbot Test")

class MedicalBot:
    def __init__(self):
        openai_model="gpt-3.5-turbo"
        st.session_state["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

        DB_FAISS_PATH = './vectorstores/db_faiss'
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cuda'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings)
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        llm = ChatOpenAI(model=openai_model, temperature=0.8, streaming=True)
        self.chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
            ),
        memory=memory
        )

    @utils.enable_chat_history
    def main(self):
        user_query=st.chat_input(placeholder="Enter Query", key="user_query")
        if user_query:
            utils.display_message( 'user', user_query)
            with st.chat_message("assistant"):
                print("Started here")
                with st.spinner("Thinking..."):
                    st_cb = StreamHandler(st.empty())
                    print("user_query", user_query)
                    response = self.chain.run(user_query, callbacks=[st_cb])
                # response = self.chain({"query": user_query})
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = MedicalBot()
    obj.main()

