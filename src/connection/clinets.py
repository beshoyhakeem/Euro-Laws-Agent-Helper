from openai import AzureOpenAI
import weaviate
from dotenv import load_dotenv
import os

load_dotenv('.env.example/.env')

# Azure OpenAI Keys
AZURE_API_KEY = os.environ["AZURE_API_KEY"]
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]

# weaviate Keys
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

model_name = "text-embedding-3-small"

# Azure OpenAi clinets
embed_client = AzureOpenAI(
    api_key= AZURE_API_KEY,
    api_version="2024-12-01-preview",
    azure_deployment = "text-embedding-3-small",
    azure_endpoint= AZURE_ENDPOINT
)

chat_client = AzureOpenAI(
    api_key = AZURE_API_KEY,
    api_version ="2025-01-01-preview",
    azure_deployment = "gpt-4.1",
    azure_endpoint = AZURE_ENDPOINT
)

# Weaviate clinets
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=WEAVIATE_API_KEY,
)