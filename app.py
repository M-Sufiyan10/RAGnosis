import os

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from data_cleaning import load_documents,prepare_documents_with_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_community.llms import Together
import streamlit as st 
import time
load_dotenv()
from together import Together

# Set up API keys

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("Gemini api key not found!!")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    
# Initialize LangChain ChromaDB with Google Embeddings
st.sidebar.title("Settings")
chunk_size=st.sidebar.slider("Chunk Size", min_value=100, max_value=1000, value=500, key="chunk_size_slider")
chunk_overlap=st.sidebar.slider("Chunk Overlap", min_value=10, max_value=100, value=25, step=5, key="chunk_overlap_slider")
num_chunks = st.sidebar.slider("Number of chunks", min_value=1, max_value=10, value=3, step=1, key="top_k_chunks_slider")

def chunk_documents(docs_with_meta, chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunked_docs = []
    for idx, doc in enumerate(docs_with_meta):
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            # Create a unique ID for each chunk
            chunk_id = f"{doc['metadata']['source_file']}_chunk_{idx}_{i}"
            
            # Wrap the chunk in a Document object
            chunked_docs.append(
                Document(
                    page_content=chunk,  # The actual chunk text
                    metadata=doc["metadata"],  # The metadata
                    id=chunk_id  # Unique ID for each chunk
                )
            )
    return chunked_docs


def add_docs_to_chroma(docs):
    """Processes a PDF file, splits it into chunks, and adds them to ChromaDB."""
    metadata_docs=prepare_documents_with_metadata(docs)
    time.sleep(15)
    st.write("The process usualyy takes time")
    documents = chunk_documents(metadata_docs)
    time.sleep(40)
    st.write("Hang on a minute")
    vector_db.add_documents(documents)
    time.sleep(20)
    st.write("Almost there")
    print(f"Added {len(documents)} chunks to ChromaDB.")

def ask_gemini(query, k=3):
    """Uses Gemini 2.0 Flash to answer a query based on retrieved context."""
    query_vector = embedding_function.embed_query(query)
    docs = vector_db.similarity_search_by_vector_with_relevance_scores(query_vector, k=k)
    
    # Combine retrieved documents into context
    context = "".join([doc[0].page_content for doc in docs])
    print(context)
    # Define custom prompt
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer,then generalized the answer based on your knwoledge
    
    Context:
    {context}
    
    Answer the following question. Let's think step by step.
    
    Question: {query}
    """
    prompt = template.format(context=context, query=query)
    
    # Invoke Gemini model with formatted prompt
    #response = client.chat.completions.create(
    #model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    #messages=[{"role": "AI", "content": prompt}])
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
    response = llm.invoke(prompt)
    
    return response
# Example Usage
if __name__ == "__main__":
    st.title("AI RAGNOSIS")
    file_path = "Finished" 
    query = st.text_input("Ask a question (or type 'exit' to quit): ", key="query_input")
    
    if st.button("Submit", key="submit_button" and query != "exit"):
        docs = load_documents(file_path) 
        add_docs_to_chroma(docs)
    elif  query == "exit":
        vector_db.delete_collection()
        st.write("Session ended. Database cleared.")

    
    if query and query.lower() != 'exit':
        answer = ask_gemini(query, k=num_chunks)
        st.header("AI RESPONSE")
        st.write(answer)
        
        if st.button("Clear", key="clear_button"):
            st.rerun()
    elif query and query.lower() == 'exit':
        vector_db.delete_collection()
        st.write("Session ended. Database cleared.")
        

