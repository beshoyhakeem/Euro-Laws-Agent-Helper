from src.core.resources import get_laws, get_vectorstore , weaviate_client
import json


laws = get_laws()
vectorstore = get_vectorstore()


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
"""
vectorstore = WeaviateVectorStore(
    client = weaviate_client,
    index_name = "Euro_Laws",
    text_key="text",
    embedding = embedding_model
)
"""

def search_docs(query):
    celex_ids = []
    full_doc_info = """ """

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    for doc in docs:
        celex_ids.append(doc.metadata['celex'])

    celex_ids = list(set(celex_ids))    

    for i , celex_id in enumerate(celex_ids):

        full_doc_info += f"""

        doc {i} :

        'celex': {laws[laws['CELEX'] == celex_id]['CELEX'].iloc[0]}
        'status': {laws[laws['CELEX'] == celex_id]['Status'].iloc[0]}
        'act_type': {laws[laws['CELEX'] == celex_id]['Act_type'].iloc[0]}
        'treaty': {laws[laws['CELEX'] == celex_id]['Treaty'].iloc[0]}

        full_doc :

        {laws[laws['CELEX'] == celex_id]['act_raw_text'].iloc[0]}
{"=="*15} "END OF DOC" {"=="*15}
        """

    return full_doc_info    



if __name__ == "__main__":

    print ("true")

