from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # type:ignore
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings  # type:ignore
from chromadb.config import Settings  # type:ignore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
import os
import json
import hashlib
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

print(f"[rag_system.py] GROQ_API_KEY loaded: {os.getenv('GROQ_API_KEY') is not None}")

# Configuration
CHROMA_DIR = "chroma_store"
DOCUMENTS_DIR = "documents"
COLLECTION_NAME = "customer_support_kb"
DOCUMENT_TRACKER_FILE = os.path.join(CHROMA_DIR, "document_tracker.json")

# ChromaDB settings
CHROMA_SETTINGS = Settings(persist_directory=CHROMA_DIR, anonymized_telemetry=False)

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
)

# System prompt for customer support
SYSTEM_PROMPT = """You are a helpful and knowledgeable customer support AI assistant.
You are part of a Retrieval-Augmented Generation (RAG) system and must answer questions based on the company's documentation.
Provide a concise and direct answer to the user's question. If the provided context is lengthy, summarize the key points relevant to the question.
If the documentation doesn't contain the answer, respond politely that you couldn't find the information.
Always be professional, clear, and concise in your responses.
Do not make up answers or provide information not in the documentation."""


def get_file_hash(filepath):
    """Calculate MD5 hash of a file to detect changes."""
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None


def load_document_tracker():
    """Load the document tracking information."""
    if os.path.exists(DOCUMENT_TRACKER_FILE):
        try:
            with open(DOCUMENT_TRACKER_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_document_tracker(tracker):
    """Save the document tracking information."""
    os.makedirs(os.path.dirname(DOCUMENT_TRACKER_FILE), exist_ok=True)
    with open(DOCUMENT_TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


def get_new_or_modified_documents():
    """Get list of new or modified documents since last processing."""
    tracker = load_document_tracker()
    documents_to_process = []

    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        print(f"Created {DOCUMENTS_DIR} directory. Please add your documents there.")
        return documents_to_process

    supported_extensions = {".txt", ".pdf", ".docx", ".doc"}

    for filename in os.listdir(DOCUMENTS_DIR):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in supported_extensions or not os.path.isfile(filepath):
            continue

        current_hash = get_file_hash(filepath)
        if not current_hash:
            continue

        file_key = f"{filename}_{os.path.getsize(filepath)}"

        # Check if file is new or modified
        if file_key not in tracker or tracker[file_key]["hash"] != current_hash:
            documents_to_process.append(filepath)

    return documents_to_process


def load_documents(filepaths=None):
    """Load specific documents or all documents if filepaths is None."""
    documents = []

    if filepaths is None:
        # Load all documents (for initial setup)
        if not os.path.exists(DOCUMENTS_DIR):
            return documents

        supported_extensions = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
        }

        for filename in os.listdir(DOCUMENTS_DIR):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext in supported_extensions:
                try:
                    loader = supported_extensions[ext](filepath)
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    print(f"Loaded: {filename} ({len(loaded_docs)} chunks)")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    else:
        # Load specific documents
        supported_extensions = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
        }

        for filepath in filepaths:
            ext = os.path.splitext(filepath)[1].lower()
            filename = os.path.basename(filepath)

            if ext in supported_extensions:
                try:
                    loader = supported_extensions[ext](filepath)
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    print(f"Loaded: {filename} ({len(loaded_docs)} chunks)")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    return documents


def chunk_documents(documents):
    """Split documents into chunks for processing."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_documents(documents)


def initialize_vectorstore():
    """Initialize or update the vector store with only new/modified documents."""
    tracker = load_document_tracker()

    # Check if vector store already exists
    vectorstore_exists = os.path.exists(CHROMA_DIR) and any(
        fname.endswith(".parquet") for fname in os.listdir(CHROMA_DIR)
    )

    if vectorstore_exists:
        print("Existing vector store found. Checking for document updates...")
        new_documents = get_new_or_modified_documents()

        if not new_documents:
            print("No new or modified documents found. Using existing vector store.")
            return get_vectorstore()

        print(f"🔄 Found {len(new_documents)} new or modified documents to process")
        documents = load_documents(new_documents)
    else:
        print("Initializing new vector store...")
        documents = load_documents()

    if not documents:
        print("No documents found. Using empty knowledge base.")
        # Create empty vector store
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
        )
    else:
        chunks = chunk_documents(documents)
        print(f"Chunked into {len(chunks)} pieces")

        if vectorstore_exists:
            # Add to existing vector store
            vectorstore = get_vectorstore()
            vectorstore.add_documents(chunks)
            print("Added new documents to existing vector store")
        else:
            # Create new vector store
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
                collection_name=COLLECTION_NAME,
            )
            print("New vector store created")

    # Update document tracker
    update_document_tracker()
    return vectorstore


def update_document_tracker():
    """Update the document tracker with current file states."""
    tracker = load_document_tracker()

    if not os.path.exists(DOCUMENTS_DIR):
        return

    supported_extensions = {".txt", ".pdf", ".docx", ".doc"}

    for filename in os.listdir(DOCUMENTS_DIR):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext not in supported_extensions or not os.path.isfile(filepath):
            continue

        file_hash = get_file_hash(filepath)
        if file_hash:
            file_key = f"{filename}_{os.path.getsize(filepath)}"
            tracker[file_key] = {
                "hash": file_hash,
                "filename": filename,
                "last_processed": datetime.now().isoformat(),
                "size": os.path.getsize(filepath),
            }

    save_document_tracker(tracker)
    print("Document tracker updated")


def get_vectorstore():
    """Get the vector store instance."""
    try:
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
        )
        # Test connection
        vectorstore._collection.count()
        return vectorstore
    except:
        # Initialize if not exists
        return initialize_vectorstore()


def rag_search(query: str, conversation_history=None):
    """
    Search for answers using RAG system.
    Returns answer string or None if no relevant information found.
    """
    try:
        vectorstore = get_vectorstore()

        # Check if we have any documents
        if vectorstore._collection.count() == 0:
            return None

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},  # Get top 3 most relevant chunks
        )

        # Initialize LLM
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,  # Lower temperature for more factual responses
            model_name="llama-3.3-70b-versatile",
        )

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Question: {question}\n\nContext: {context}"),
            ]
        )

        # Use simple retrieval for better control
        relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            return None

        # Combine context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Create formatted prompt
        formatted_prompt = prompt.format_messages(question=query, context=context)

        # Get response from LLM
        response = llm.invoke(formatted_prompt)

        # Check if the response indicates no information found
        response_text = response.content.strip()
        if any(
            phrase in response_text.lower()
            for phrase in [
                "not in the documentation",
                "couldn't find",
                "don't have that information",
                "not contained",
            ]
        ):
            return None

        return response_text

    except Exception as e:
        print(f"RAG search error: {e}")
        return None


def get_knowledge_base_stats():
    """Get statistics about the knowledge base."""
    try:
        vectorstore = get_vectorstore()
        count = vectorstore._collection.count()
        return {
            "document_count": count,
            "collection_name": COLLECTION_NAME,
            "persist_directory": CHROMA_DIR,
        }
    except:
        return {"document_count": 0, "status": "not_initialized"}


# Initialize on import
if not os.path.exists(CHROMA_DIR):
    os.makedirs(CHROMA_DIR)

# Quick test to ensure everything works
if __name__ == "__main__":
    print("Testing RAG system...")
    stats = get_knowledge_base_stats()
    print(f"Knowledge base stats: {stats}")
