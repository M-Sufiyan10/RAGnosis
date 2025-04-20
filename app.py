__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from data_cleaning import load_documents,prepare_documents_with_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
import streamlit as st 
from langchain_community.vectorstores import FAISS
load_dotenv()

# Set up API keys

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("Gemini api key not found!!")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY
print(GOOGLE_API_KEY)

embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)
vector_db = None
    
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


def add_docs_to_faiss(docs):
    """Processes a PDF file, splits it into chunks, embeds them, and adds to FAISS."""
    global vector_db
    metadata_docs = prepare_documents_with_metadata(docs)
    
    st.write("The process usually takes time")

    # Chunking
    documents = chunk_documents(metadata_docs)
    st.success("successfully chunked documents")
    texts = [doc.page_content for doc in documents]
    embeddings=embedding_function.embed_documents(texts)
    embeddings_pairs=zip(texts,embeddings)
    # Let FAISS handle embeddings internally
    st.session_state.vector_db = FAISS.from_embeddings(embeddings_pairs,embedding_function
    )
    st.success(f"Added {len(documents)} chunks to FAISS DB each of 768 dimensions")


def ask_gemini(query, k=3):
    """Uses Gemini 1.5 Flash to answer a query based on retrieved context."""
    query_vector = embedding_function.embed_query(query)
    vector_db = st.session_state.vector_db
    docs = vector_db.similarity_search_with_score_by_vector(query_vector, k=k)
    
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
    
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
    response = llm.invoke(prompt)
    
    return response


# Example Usage
if __name__ == "__main__":
    st.title("AI RAGNOSIS")
    
# Load and embed documents once
    if "vector_db_initialized" not in st.session_state:
        st.write("Wait a few minutes while the system loads the documents...")
        docs = load_documents("Finished")
        add_docs_to_faiss(docs)
        st.session_state.vector_db_initialized = True
        st.success("Documents loaded and embedded successfully!")

    # Accept user query input
    query = st.text_input("Ask a question:", key="query_input")
    # Process query
    if query:
        if query.lower() == "exit":
            vector_db.delete()
            st.session_state.vector_db_initialized = False
            st.write("Session ended. Vector DB cleared.")
        else:
            answer = ask_gemini(query, k=num_chunks)
            st.header("AI RESPONSE")
            st.write(answer)


        
