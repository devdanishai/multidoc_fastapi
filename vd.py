from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

# loaidng the embedding model
embeddings = HuggingFaceEmbeddings()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

# Initialize DirectoryLoader with the path and loader class
directory_loader = DirectoryLoader(
    path="data",           # Your directory containing PDF files
    glob="*.pdf",          # Pattern to match PDF files
    loader_cls=PyPDFLoader  # Specify PyPDFLoader as the loader class
)

# Load all documents in the specified directory
documents = directory_loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2000,
                                      chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)

print("Documents Vectorized")