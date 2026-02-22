import pandas as pd
from src.connection.clinets import weaviate_client
from src.sentencetransformers.st_class import SentenceTransformersEmbeddings
from langchain_weaviate import WeaviateVectorStore

# Private globals (initially None)
_laws = None
_embedding_model = None
_vectorstore = None


def get_laws():
    global _laws
    if _laws is None:
        print("Loading laws CSV...")
        _laws = pd.read_csv(
    "dataset/act_raw_text_with_4meta.csv",
    usecols=["CELEX", "Status", "Act_type", "Treaty", "act_raw_text"],
    low_memory=True,
)
    return _laws


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformersEmbeddings(
            "sentence-transformers/all-mpnet-base-v2"
        )
    return _embedding_model


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("Initializing vector store...")
        _vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name="Euro_Laws",
            text_key="text",
            embedding=get_embedding_model(),
        )
    return _vectorstore


def close_connections():
    print("Closing Weaviate connection...")
    weaviate_client.close()