################add multiple URLs ################################

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_vectorstore_from_urls(urls):
    all_document_chunks = []
    for url in urls:
        loader = WebBaseLoader(url)
        document = loader.load()

        text_splitter = RecursiveCharacterTextSplitter()
        document_chunks = text_splitter.split_documents(document)
        all_document_chunks.extend(document_chunks)
    
    vector_store = Chroma.from_documents(all_document_chunks, OpenAIEmbeddings())

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613")
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# app config
st.set_page_config(page_title="Vibe Surf Guru", page_icon="🌊")
st.title("Vibe Surf Guru - J-Bay")

# List of website URLs
website_urls = [
    "https://www.surf-forecast.com/breaks/J-Bay/forecasts/latest/six_day",
    "https://www.yeeew.com/mag/surfing-jeffreys-bay-a-guide-to-one-of-the-worlds-best-waves/"
    "https://www.surfline.com/surf-news/mechanics-of-jeffreys-bay-jbay-worlds-best-right-point/28782"

]

# Initialize vector store and chat history if not already done
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore_from_urls(website_urls)
if "chat_history" not in st.session_state or not st.session_state.chat_history:
    st.session_state.chat_history = [AIMessage(content="Hello, I am Vibe Surf Guru. Ask me anything related to surfing in J-BAY?")]
if "human_message_count" not in st.session_state:
    st.session_state.human_message_count = 0  # Initialize counter

# User input
user_query = st.chat_input("Ask me about the surfspot here BRUH...")
if user_query is not None and user_query != "":
    # Increment counter and check condition before appending the message
    st.session_state.human_message_count += 1
    if st.session_state.human_message_count >= 4:  # Reset on the 4th question
        # Get response for the 4th question before resetting
        response = get_response(user_query)
        # Reset conversation with the current query as the first message
        st.session_state.chat_history = [
            HumanMessage(content=user_query),
            AIMessage(content=response)
        ]
        st.session_state.human_message_count = 1  # Reset counter with current question counted
    else:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
