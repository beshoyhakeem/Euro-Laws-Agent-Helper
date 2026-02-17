from sentence_transformers import SentenceTransformer
from typing import List

class SentenceTransformersEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # returns a list of embeddings for documents
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        # returns a single embedding for a query
        return self.model.encode([text])[0].tolist()