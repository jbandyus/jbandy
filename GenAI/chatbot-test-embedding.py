# chatbot-test-embedding.py
# run with: streamlit run chatbot-test-embedding.py
# simple script to build a UI to test the embdedded data

# Imports
import streamlit as st
from openai import OpenAI
#import glib
#from langchain_community.embeddings import OpenAIEmbeddings,ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS

# Statics
model='gpt-3.5-turbo'
embedding= 'FAISS'

st.title(f"Chat with {model} using {embedding} embeddings")

# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = model

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I assist you today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    if (embedding=='FAISS'):
        # change this to use the local FAISS vector store
        embedding_func = OpenAIEmbeddings()
        db= FAISS.load_local("./faiss_db",embedding_func,allow_dangerous_deserialization="True")
    else:
        print("ERROR: embedding not specified")        
        exit(1)
    
    # print results from a similarity search
    #docs = db.similarity_search(prompt)
    #print(docs[0].page_content)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    # Use RetrievalQA chain for orchestration
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
    response = qa.run(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    