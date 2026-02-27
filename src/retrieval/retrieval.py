from src.core.resources import get_vectorstore , weaviate_client
from weaviate.classes.query import Filter
import json

vectorstore = get_vectorstore()

# To get the full doc form Euro_Law_Documents collection
eur_docs = weaviate_client.collections.get("Euro_Law_Documents")


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
 
# to get the full doc with metadata from Euro_Law_Documents collection
def get_full_doc_weaviate(celex_id):

    response = eur_docs.query.fetch_objects(
    filters=Filter.by_property("celex").equal(celex_id),
    limit=1
)
    r = response.objects[0].properties

    # r is a dict with key, values for each doc
    return r    

################ Using LangChain ####################

# search docs using query to get a string containg all docs with metadata
def search_docs(query):
    celex_ids = []
    full_doc_info = """ """

    # Search docs using vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    for doc in docs:
        celex_ids.append(doc.metadata['celex'])

    # Remove duplicates from celex_ids list
    celex_ids = list(set(celex_ids))    

    for i , celex_id in enumerate(celex_ids):
        f_doc_meta = get_full_doc_weaviate(celex_id)

        full_doc_info += f"""

        doc {i} :

        'celex': {f_doc_meta['celex']}
        'status': {f_doc_meta['status']}
        'act_type': {f_doc_meta['act_type']}
        'treaty': {f_doc_meta['treaty']}

        full_doc :

        {f_doc_meta['full_doc']}
{"=="*15} "END OF DOC" {"=="*15}
        """

    return full_doc_info    



if __name__ == "__main__":

    print ("true")

    print(search_docs("Drug dealing Sentences"))

    

