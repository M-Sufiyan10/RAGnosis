import os
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from data_cleaning import load_documents,prepare_documents_with_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()
# Set up API keys

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if GOOGLE_API_KEY is None:
    raise ValueError("Gemini api key not found!!")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    
# Initialize LangChain ChromaDB with Google Embeddings
def chunk_documents(docs_with_meta, chunk_size=500, chunk_overlap=50):
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
    documents = chunk_documents(metadata_docs)
    vector_db.add_documents(documents)
    print(f"Added {len(documents)} chunks to ChromaDB.")

def ask_gemini(query):
    """Uses Gemini 2.0 Flash to answer a query based on retrieved context."""

    
    query_vector = embedding_function.embed_query(query)
    docs = vector_db.similarity_search_by_vector(query_vector, k=3)
    
    # Combine retrieved documents into context
    context = "".join([doc.page_content for doc in docs])
    
    # Define custom prompt
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Answer the following question. Let's think step by step.
    
    Question: {query}
    """
    prompt = template.format(context=context, query=query)
    
    # Invoke Gemini model with formatted prompt
    llm = GoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GEMINI_API_KEY"))
    response = llm.invoke(prompt)
    
    return response
# Example Usage
if __name__ == "__main__":
    file_path = "Finished" 
    docs=load_documents(file_path) 
    add_docs_to_chroma(docs)
    
    while True:
        query = input("Ask a question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            vector_db.delete_collection()
            break
        answer = ask_gemini(query)
        print("\nAI Response:\n", answer)

