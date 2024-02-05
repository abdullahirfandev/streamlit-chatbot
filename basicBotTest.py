#!/usr/bin/env python3


# Set the model and test it
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain

load_dotenv()

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstores/db_faiss'

customPromptTemplate = """use the following pieces of information to answer the user's question.
If you don't know the answer, please say that you don't know the answer, don't try to make up
an answer

Context: {}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def setCustomPrompt():
    """
    Prompt template for QA retrieval for each vector store
    """
    prompt = PromptTemplate(template=customPromptTemplate, input_variables=['context', 'question'])
    return prompt


def loadLlm():
    """
    Load the OpenAI LLM for chat-based interactions.
    """
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv('OPENAI_API_KEY'))

def retrievalQaChain(llm, prompt, db):
    qaChain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(
            search_type='mmr',
            search_kwargs={'k':2, 'fetch_k':4}
            ),
        return_source_documents=True,
    )
    return qaChain

def qaBot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = loadLlm()
    qaPrompt = setCustomPrompt()
    qa = retrievalQaChain(llm, qaPrompt, db)
    return qa

def finalResult(query):
    qaResult = qaBot()
    response = qaResult({'query': query})
    return response

def evaluateModel():
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    example_gen_chain = QAGenerateChain.from_llm(loadLlm())
    new_examples = example_gen_chain.apply_and_parse([
        {'doc': t} for t in documents[10:25]
    ])
    qa = qaBot()
    examples = [item['qa_pairs'] for item in new_examples]
    predictions = qa.apply(examples)
    llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-1106', openai_api_key=os.getenv('OPENAI_API_KEY'))
    eval_chain = QAEvalChain.from_llm(llm)
    graded_outputs = eval_chain.evaluate(examples, predictions)
    for i, eg in enumerate(examples):
        print(f"Example {i+1}:")
        print("Question: " + predictions[i]['query'])
        print("Real Answer: " + predictions[i]['answer'])
        print("Predicted Answer: " + predictions[i]['result'])
        print("Predicted Grade: " + graded_outputs[i]['results'])
        print()


# Example usage
print(finalResult("what is Eczema?"))
print(finalResult("what are it's properties?"))

# evaluateModel()