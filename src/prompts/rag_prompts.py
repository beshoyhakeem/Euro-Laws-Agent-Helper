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
        ("system", """You are a query rebuilder for semantic document retrieval from a vector database.

Your task is to transform the user's question into an optimized retrieval query.

Focus on:
- Extracting the core legal issues
- Identifying relevant legal concepts, doctrines, statutes, and terminology
- Removing conversational or irrelevant language
- Preserving factual constraints (dates, jurisdictions, parties, contract types, etc.)
- Expanding implicit legal intent into explicit searchable terms

The output should be concise, keyword-dense, and optimized for vector similarity search.
Do NOT answer the question.
Only return the enhanced retrieval query."""),
        
        ("human", """Original question: {question}

Rewritten retrieval query:"""),
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