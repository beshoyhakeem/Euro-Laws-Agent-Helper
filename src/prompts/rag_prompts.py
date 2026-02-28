from langchain_core.prompts import ChatPromptTemplate

################################## Classifier prompt #####################################

classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a classifier. Determine if the user's question requires retrieving documents from a knowledge base or can be answered directly or from previous conversation or history."),
        ("human", """Respond with ONLY "normal" or "rag" or "history":

- "normal": General questions, greetings, or questions that don't need document retrieval
- "rag": Questions about specific laws, regulations, or information that requires document search
- "history": Questions that require referencing previous conversation history

Question: {question}
Response:"""),
    ]
)

######################################## Query enhancement prompt ##################################

query_enhance_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a legal query optimizer for semantic document retrieval.

Rewrite the user's question into a concise, keyword-dense search query.

1. Identify the user's intent (e.g., definition, sentencing, elements, defenses, procedure, case law, jurisdiction).
2. Include only terms relevant to that intent.
3. If the user asks about sentence/punishment/penalty, prioritize sentencing range, statutory penalties, fines, imprisonment, criminal sanctions, and jurisdiction.
4. Do not default to definitions unless explicitly requested.
5. Remove conversational language.
6. Do NOT answer the question.

Return only the optimized retrieval query."""),
        
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