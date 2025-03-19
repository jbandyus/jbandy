# chatbot-test-embedding.py
# V.1 early release of sample code
# run with: streamlit run chatbot-test-embedding.py
# simple UI to test a prompt against a vector database
# 

# Imports
import streamlit as st
import argparse
from openai import OpenAI
#import glib
#from langchain_community.embeddings import OpenAIEmbeddings,ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.chroma import Chroma

# Statics

# Main script
if __name__ == '__main__':

    # Move this to a def
    parser = argparse.ArgumentParser()
    parser.add_argument('model',choices=['gpt-3.5-turbo'], help="Base model to chat with")
    parser.add_argument('embedding_db',choices=['ChromaDB','FAISS'], help="method used to store the embeddings")
    parser.add_argument('-d','--datadir', nargs=1,help='Directory to read the data files (local models only)')
    parser.add_argument("-v", "--verbosity", action="count", default=0, help="increase output verbosity")
    args = parser.parse_args()
    DEBUG=args.verbosity
    if (DEBUG): print(f"args: {args}")
    model= args.model
    embedding_db= args.embedding_db
    #if (embedding_func=='OPENAI' and embedding_db=='ChromaDB'):
    #    print("ERROR: OPENAI is not supported with ChromaDB")
    #    exit(1)
    if (args.datadir):
        data_dir=args.datadir[0]
    elif (embedding_db=='FAISS'):
        data_dir= "./FAISS"
    elif (embedding_db=='ChromaDB'):
        data_dir="./ChromaDB"
    if (DEBUG): print(f"DATA_DIR={data_dir}")
    if (DEBUG): print(f"DEBUG={DEBUG}")
    # stop for input testing
    #st.stop() # press control-C before stopping the browser, stop seems to not help
    #exit(0)

    st.title(f"Chat with {model} using a {embedding_db} embedding database")

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

        if (embedding_db=='FAISS'):
            # change this to use the local FAISS vector store
            embedding_func = OpenAIEmbeddings()
            db= FAISS.load_local(data_dir,embedding_func,allow_dangerous_deserialization="True")
        elif (embedding_db=='ChromaDB'):
            embedding_func = OpenAIEmbeddings()
            # This call seems to be wrong for Chroma
            #db = Chroma(persist_directory=data_dir, embedding_function=embedding_func)
            print("ERROR: ChromaDB database currently not working")        
            exit(1)
        else:
            print("ERROR: embedding database not specified")        
            exit(1)
    
        # print results from a similarity search
        if (DEBUG>1): 
            docs = db.similarity_search(prompt)
            print(docs[0].page_content)
        # extra functions for use later in another script
        #
        # load from disk
        #db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        #docs = db3.similarity_search(query)
        #print(docs[0].page_content)
        # load it into Chroma
        #db = Chroma.from_documents(docs, embedding_function)
        # query it
        #query = "What is being done to protect the wild wolves of North America?"
        #docs = db.similarity_search(query)
        # print results

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
        # Use RetrievalQA chain for orchestration
        qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
        response = qa.run(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    