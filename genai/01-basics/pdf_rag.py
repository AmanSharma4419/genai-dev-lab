import json
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI

client = OpenAI()

# Load PDF
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "pdf_file" / "resume.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(docs)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector DB 
vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    url="http://localhost:6333",
    collection_name="pdf-files",
    embedding=embeddings
)

# Prompt template
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

while True:
    user_inp = input("\n> ")

    # 1. Retrieve relevant docs
    results = vector_store.max_marginal_relevance_search(
    user_inp,
    k=10,
    fetch_k=20,
    lambda_mult=0.5
)
    # 2. Convert docs → text
    context = "\n\n".join([doc.page_content for doc in results])
    # 3. Inject into prompt
    formatted_prompt = SYSTEM_PROMPT.format(context=context)

    # 4. Send to LLM
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_inp}
        ]
    )

    # 5. Print response
    print("\nAI:", response.choices[0].message.content)