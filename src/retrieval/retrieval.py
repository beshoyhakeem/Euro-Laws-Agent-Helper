from typing import Any

from src.connection.clinets import weaviate_client
from src.sentencetransformers.st_class import SentenceTransformersEmbeddings
from langchain_weaviate import WeaviateVectorStore

import json
import pandas as pd

laws = pd.read_csv('dataset/Celex_act_raw_text.csv')

embedding_model = SentenceTransformersEmbeddings('sentence-transformers/all-mpnet-base-v2')

def search_chunks(query_embedding):

    # Use Euro_Laws collection
    Euro_Laws = weaviate_client.collections.use("Euro_Laws")

    # Perform a vector search with NearVector
    response = Euro_Laws.query.near_vector(
        near_vector= query_embedding, 
        limit=5
    )

    # Retrive nerest chunks with metadata and put in list 
    retrived = []
    for obj in response.objects:
        retrived.append(json.dumps(obj.properties))

    # Get the celex id for nerest chunks
    celex_ids = []
    for meta in retrived:
        key_value = json.loads(meta)
        celex_ids.append(key_value.get("celex", ""))

    return retrived , celex_ids


# Get The Full Docs Using celex_id
def get_docs_celex(celex_ids):
    docs_list = []

    # to remove the duplicates 
    celex_ids = list(set(celex_ids))

    for celex_id in celex_ids:
        doc = laws[laws['CELEX'] == celex_id]['act_raw_text'].iloc[0]
        docs_list.append(doc)

    return docs_list 

################ Using LangChain ####################

# intialize vector store
vectorstore = WeaviateVectorStore(
    client = weaviate_client,
    index_name = "Euro_Laws",
    text_key="text",
    embedding = embedding_model
)


def lang_search_chunks(user_question):
    celex_ids = []

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(user_question)

    for doc in docs:
        celex_ids.append(doc.metadata['celex'])

    celex_ids = list[Any](set(celex_ids))

    return docs , celex_ids 

if __name__ == "__main__":

    print ("true")

    docr , celex_id = lang_search_chunks("Pooping in Public Places")

    print(celex_id)
