import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from neo4j import GraphDatabase

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient

from qdrant_client.models import VectorParams, Distance


# =========================================================
# OPENAI
# =========================================================

llm_client = OpenAI()


# =========================================================
# QDRANT SETUP
# =========================================================

COLLECTION_NAME = "storybook-rag"

qdrant_client = QdrantClient(
    url="http://localhost:6333"
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

collections = qdrant_client.get_collections().collections

exists = any(
    c.name == COLLECTION_NAME
    for c in collections
)

if not exists:

    print("\nCreating Qdrant collection...\n")

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=3072,
            distance=Distance.COSINE
        )
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)


# =========================================================
# NEO4J SETUP
# =========================================================

neo4j_driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password")
)


# =========================================================
# PDF LOAD
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "pdf_file" / "riding-hood.pdf"

loader = PyPDFLoader(file_path)

docs = loader.load()


# =========================================================
# CHUNKING
# =========================================================

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100
)

split_docs = text_splitter.split_documents(docs)


# =========================================================
# CLEAN CYPHER
# =========================================================

def clean_cypher(text: str):

    text = text.replace("```cypher", "")
    text = text.replace("```", "")

    return text.strip()


# =========================================================
# GENERATE CYPHER FOR GRAPH STORAGE
# =========================================================

def generate_cypher_from_text(text):

    prompt = f"""

You are an expert Neo4j knowledge graph generator.

Convert the given story text into Neo4j Cypher queries.

IMPORTANT RULES:
- Return ONLY raw Cypher
- No markdown
- No explanations
- Use MERGE only
- Create relationships carefully

Allowed Labels:
- Character
- Place
- Animal
- Object

Allowed Relationships:
- MET
- VISITED
- TALKED_TO
- ATE
- HELPED
- FOLLOWED
- WARNED
- LIVED_IN

Examples:

Story:
Red Riding Hood met the wolf.

Cypher:
MERGE (r:Character {{name:'Red Riding Hood'}})
MERGE (w:Animal {{name:'Wolf'}})
MERGE (r)-[:MET]->(w)


Story:
The wolf ate grandmother.

Cypher:
MERGE (w:Animal {{name:'Wolf'}})
MERGE (g:Character {{name:'Grandmother'}})
MERGE (w)-[:ATE]->(g)


Now generate Cypher.

Story:
{text}

"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw_cypher = response.choices[0].message.content

    return clean_cypher(raw_cypher)


# =========================================================
# EXECUTE CYPHER
# =========================================================

def execute_cypher(cypher_text):

    queries = [
        q.strip()
        for q in cypher_text.split("\n")
        if q.strip()
    ]

    with neo4j_driver.session() as session:

        for query in queries:

            try:

                session.run(query)

                print("\nExecuted:")
                print(query)

            except Exception as e:

                print("\nFAILED QUERY:")
                print(query)

                print("\nERROR:")
                print(e)


# =========================================================
# INDEX PDF
# =========================================================

print("\nIndexing PDF...\n")

for doc in split_docs:

    # -----------------------------------------------------
    # STORE EMBEDDINGS IN QDRANT
    # -----------------------------------------------------

    vector_store.add_documents([doc])

    # -----------------------------------------------------
    # GENERATE CYPHER
    # -----------------------------------------------------

    cypher = generate_cypher_from_text(
        doc.page_content
    )

    print("\nGenerated Cypher:")
    print(cypher)

    # -----------------------------------------------------
    # STORE GRAPH IN NEO4J
    # -----------------------------------------------------

    execute_cypher(cypher)

print("\nPDF INDEXING COMPLETED.\n")


# =========================================================
# GENERATE CYPHER FOR USER QUESTION
# =========================================================

def generate_query_cypher(question):

    prompt = f"""

You are an expert Neo4j Cypher generator.

Convert the user question into a valid Cypher query.

IMPORTANT RULES:
- Return ONLY raw Cypher
- No markdown
- No explanations

==================================================
GRAPH SCHEMA
==================================================

Labels:
- Character
- Place
- Animal
- Object

Relationships:
- MET
- VISITED
- TALKED_TO
- ATE
- HELPED
- FOLLOWED
- WARNED
- LIVED_IN

==================================================
EXAMPLES
==================================================

Question:
Who did Red Riding Hood meet?

Cypher:
MATCH (r:Character {{name:'Red Riding Hood'}})
-[:MET]->
(o)
RETURN o.name


Question:
Who ate grandmother?

Cypher:
MATCH (w)-[:ATE]->(g {{name:'Grandmother'}})
RETURN w.name


Question:
Who talked to the wolf?

Cypher:
MATCH (p)-[:TALKED_TO]->(w {{name:'Wolf'}})
RETURN p.name


==================================================
NOW GENERATE CYPHER
==================================================

Question:
{question}

"""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw_cypher = response.choices[0].message.content

    return clean_cypher(raw_cypher)


# =========================================================
# QUERY GRAPH
# =========================================================

def query_graph(cypher):

    with neo4j_driver.session() as session:

        result = session.run(cypher)

        return [dict(record) for record in result]


# =========================================================
# FINAL CHAT PROMPT
# =========================================================

SYSTEM_PROMPT = """

You are a helpful AI assistant.

Use:
1. Graph relationship data
2. Story document chunks

to answer the user.

If answer is not found:
say "I don't know"

"""


# =========================================================
# CHAT LOOP
# =========================================================

while True:

    user_input = input("\n> ")

    if user_input.lower() == "exit":
        break

    # QDRANT SEARCH

    vector_results = vector_store.similarity_search(
        user_input,
        k=4
    )

    vector_context = "\n\n".join([
        doc.page_content
        for doc in vector_results
    ])

    # CYPHER GENERATION

    cypher_query = generate_query_cypher(
        user_input
    )

    print("\nGenerated Cypher:")
    print(cypher_query)

    # GRAPH QUERY

    try:

        graph_results = query_graph(
            cypher_query
        )

    except Exception as e:

        print("\nGraph Query Failed:")
        print(e)

        graph_results = []

    graph_context = json.dumps(
        graph_results,
        indent=2
    )

    # FINAL CONTEXT

    final_context = f"""

GRAPH DATA:
{graph_context}

DOCUMENT DATA:
{vector_context}

"""

    # FINAL GPT ANSWER

    response = llm_client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "system",
                "content": final_context
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
    )

    print("\nAI:\n")

    print(response.choices[0].message.content)