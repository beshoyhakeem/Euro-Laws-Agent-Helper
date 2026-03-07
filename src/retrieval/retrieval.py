from src.core.resources import get_vectorstore, get_vectorstore_full_doc, weaviate_client , get_embedding
from src.core.utils import count_tokens
from weaviate.classes.query import Filter
import json

vectorstore = get_vectorstore()

vectorstore_full_doc = get_vectorstore_full_doc()

# To get the full doc form Euro_Law_Documents collection
eur_docs = weaviate_client.collections.get("Euro_Law_Documents")

# To use the hybrid search form the collections
eur_docs_hybrid = weaviate_client.collections.get("Euro_Laws_hybrid")


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
def search_docs_full(query):
    celex_ids = []
    full_doc_info = """ """

    # for debugging purpose
    weaviate_client.connect()

    # Search docs using vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
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
        # for debugging purpose
        if len(full_doc_info) > 5:
            print("docs found and retrieved successfully")
    print(len(full_doc_info))

    weaviate_client.close()      

    return full_doc_info    


############################################### Use retrived chunks insted of the full doc ################################################

def search_chunks(query):
    full_chunks_info = """ """

    # for debugging purpose
    weaviate_client.connect()

    # Search docs using vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    for i , chunk in enumerate(docs):

        full_chunks_info += f"""

        chunk {i} :

        'celex': {chunk.metadata['celex']}
        'status': {chunk.metadata['status']}
        'act_type': {chunk.metadata['act_type']}
        'treaty': {chunk.metadata['treaty']}

        full_chunk :

        {chunk.page_content}
{"=="*15} "END OF DOC" {"=="*15}
        """
        # for debugging purpose
        if len(full_chunks_info) > 5:
            print("docs found and retrieved successfully")
    print(len(full_chunks_info))

    weaviate_client.close()      

    return full_chunks_info    


############################################### Use retrived chunks insted of the full doc with hybrid search ################################################
def search_docs_hybrid(query):
    full_chunks_info = """ """

    # for debugging purpose
    weaviate_client.connect()

    # embed query
    embed_query = get_embedding(query)

    # Search docs using vectorstore
    response = eur_docs_hybrid.query.hybrid(
        query= query,
        vector= embed_query,
        alpha=0.5,
        limit=5
    )

    for i , obj in enumerate(response.objects):

        full_chunks_info += f"""

        chunk {i} :

        'celex': {obj.properties['celex']}
        'status': {obj.properties['status']}
        'act_type': {obj.properties['act_type']}
        'treaty': {obj.properties['treaty']}

        full_chunk :

        {obj.properties['text']}
{"=="*15} "END OF DOC" {"=="*15}
        """
        # for debugging purpose
        if len(full_chunks_info) > 5:
            print("chunks found and retrieved successfully")
    print(len(full_chunks_info))
    weaviate_client.close()

    # for  for debugging purpose
    print(f"full_chunks_info : \n {full_chunks_info}")   
      

    return full_chunks_info  


############################################### Use retrived chunks celex to get full doc search to summrize it ################################################
def get_full_docs_celex(celex_ids):
    full_doc_info = """ """

    # open weaviate client
    weaviate_client.connect()

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
        # for debugging purpose
        if len(full_doc_info) > 5:
            print("docs found and retrieved successfully")
    print(len(full_doc_info))
    print(count_tokens(full_doc_info))

    #open weaviate client
    weaviate_client.close()      

    return full_doc_info



if __name__ == "__main__":

    print ("true")

    #print(search_docs_hybrid("Drug dealing Sentences"))

    print(get_full_docs_celex(["32010R0330", "32014R0316", "32010R1218"]))


