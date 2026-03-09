from src.retrieval.retrieval import get_full_docs_celex
from src.core.utils import count_tokens

def chunk_full_docs_for_summrize(celex_ids):

    chunks : list[list] = []

    full_docs, docs_list = get_full_docs_celex(celex_ids)

    if count_tokens(full_docs) > 20000:

        # chunk each doc into a list of chunks
        for doc in docs_list:

            if count_tokens(doc) < 20000:
                chunks.append([doc])

            else:

                doc_tokens_count = count_tokens(doc)
                words_in_doc = doc.split()
                no_words_doc = len(words_in_doc)

                split_coefficient: float = no_words_doc / doc_tokens_count
                split_size: int = int(19000 * split_coefficient)

                # For debugging purpuse
                print(f"doc_tokens_count: {doc_tokens_count}\nno_words_doc: {no_words_doc}\nsplit_coefficient: {split_coefficient}\nsplit_size: {split_size}")

                chunks.append([" ".join(words_in_doc[i:i+split_size]) for i in range(0, no_words_doc, split_size)])
    
    # need to put else to return the full_doc if no need for summrize

    return {"chunks_for_summrize": chunks }



