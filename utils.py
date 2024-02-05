import os
import streamlit as st


# Defining Decoder
def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):
        # clear history after switching chatbot
        current_page = func.__qualname__
        print(func.__qualname__)
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you today?"}]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])
    def execute(*args, **kwargs):
        return func(*args, *kwargs)
    return execute

def display_message(author, message):
    st.session_state.messages.append({"role": author, "content": message})
    st.chat_message(author).write(message)

