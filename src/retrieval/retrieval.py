from src.connection.clinets import *

import weaviate

import json
import pandas as pd

laws = pd.DataFrame

def search_chunks(query_embedding):
    with weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=WEAVIATE_API_KEY,
) as client:
        
        # Step 2.2: Use this collection
        Euro_Laws = client.collections.use("Euro_Laws")

        # Step 2.3: Perform a vector search with NearVector
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


def get_docs_celex(celex_ids):
    docs_list = []

    for celex_id in celex_ids:
        doc = laws[laws['CELEX'] == celex_id]['act_raw_text'].iloc[0]
        docs_list.append(doc)

    return docs_list    



if __name__ == "__main__":

    print ("true")