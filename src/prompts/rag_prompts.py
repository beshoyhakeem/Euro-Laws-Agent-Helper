from langchain_core.prompts import ChatPromptTemplate

################################## Classifier prompt #####################################

classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a classifier. Determine if the user's question requires retrieving documents from a knowledge base or can be answered directly."),
        ("human", """Respond with ONLY "normal" or "rag":

- "normal": General questions, greetings, or questions that don't need document retrieval
- "rag": Questions about specific laws, regulations, or information that requires document search

Question: {question}
Response:"""),
    ]
)

######################################## Query enhancement prompt ##################################
query_enhance_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a query enhancement assistant."),
        ("human", """Rewrite the user's question to be more effective for document retrieval.
Make it specific and include relevant legal terms.

Original question: {question}
Enhanced query:"""),
    ]
)

################################### Normal answer prompt ####################################
normal_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant for European laws."),
        ("human", """Answer the user's question directly and concisely.

Question: {question}
Answer:"""),
    ]
)

####################################### RAG answer prompt #########################################
rag_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a legal assistant specializing in European laws.
Answer the user's question using ONLY the provided context documents.
If the answer isn't in the context, say so clearly."""),
        ("human", """Context documents:
{context}

Question: {question}
Answer:"""),
    ]
)