from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

text = "Hello world!"

response = client.embeddings.create(
    input=text,
    model="text-embedding-3-small"
)

print(response)

