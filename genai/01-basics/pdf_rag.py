from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "pdf_file" / "tourism.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

split_docs = text_splitter.split_documents(documents=docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_1 = embeddings.embed_query(split_docs[0].page_content)

