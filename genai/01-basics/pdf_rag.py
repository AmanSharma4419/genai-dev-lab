import json
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from openai import OpenAI

# client setup
llm_client = OpenAI()
qdrant_client = QdrantClient(url="http://localhost:6333")

# coll naming
COLLECTION_NAME = "pdf-files"

# Loading pdf
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "pdf_file" / "resume.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

# chunking the pdf file 
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100
)

split_docs = text_splitter.split_documents(docs)

# Add metadata
for i, doc in enumerate(split_docs):
    doc.metadata.update({
        "chunk_id": i,
        "source": "resume.pdf",
        "page": doc.metadata.get("page", None)
    })

# creating embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# creation of collections
def create_collection_if_not_exists(client, collection_name):
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        print("🆕 Creating collection...")

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=3072,  
                distance=Distance.COSINE
            )
        )

        print(" Collection created.")

create_collection_if_not_exists(qdrant_client, COLLECTION_NAME)

# vector db storing qdb
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

# indexing
def index_documents_if_needed(vector_store, docs):

    vector_store.add_documents(docs)
    print("Indexing complete.")
   

index_documents_if_needed(
    vector_store,
    split_docs,   
)

# prompting
SYSTEM_PROMPT = """
You are a precise AI assistant.

Use ONLY the provided context to answer.
If the answer is not clearly in the context, say "I don't know".

Guidelines:
- Be concise and factual
- Do not hallucinate
- Prefer exact phrases from context

Context:
{context}
"""

# chat 
while True:
    user_inp = input("\n> ")

    # 🔹 Retrieve
    results = vector_store.max_marginal_relevance_search(
        user_inp,
        k=5,
        fetch_k=20,
        lambda_mult=0.5
    )

    # 🔹 Build Context
    context = "\n\n".join([doc.page_content for doc in results])

    formatted_prompt = SYSTEM_PROMPT.format(context=context)

    # 🔹 LLM Call
    response = llm_client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_inp}
        ]
    )

    # 🔹 Output
    print("\nAI:", response.choices[0].message.content)