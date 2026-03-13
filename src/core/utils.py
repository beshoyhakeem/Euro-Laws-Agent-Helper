import tiktoken

# inzlize object enc with gpt-4o model
enc = tiktoken.encoding_for_model("gpt-4o")

# token counter
def count_tokens(text):

    tokens_count = len(enc.encode(text))
    return tokens_count
