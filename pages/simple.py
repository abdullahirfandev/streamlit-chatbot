from streaming import StreamHandler
import utils
import os
import streamlit as st
# from langchain_community.llms import openai
# from langchain.llms import openai
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory



st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header("Simple Chatbot Test")

class Simple:
    def __init__(self):
        self.openai_model="gpt-3.5-turbo"
        st.session_state["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
        llm = ChatOpenAI(temperature=0, streaming=True)
        self.chain =  ConversationChain(
            llm=llm, verbose=True, memory=ConversationBufferMemory()
        )

    @utils.enable_chat_history
    def main(self):
        user_query=st.chat_input(placeholder="Enter Query", key="user_query")
        if user_query:
            utils.display_message( 'user', user_query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st_cb = StreamHandler(st.empty())
                    print("user_query", user_query)
                    response = self.chain.run(input=user_query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = Simple()
    obj.main()

