from langchain_openai import AzureChatOpenAI
from src.connection.clinets import AZURE_API_KEY , AZURE_ENDPOINT, chat_client


def message_chat(messages):
    chat_response = chat_client.chat.completions.create(
        model="gpt-4.1",
        messages= messages ,
        temperature=1
        )
    response_text = chat_response.choices[0].message.content
    
    return  response_text

"""
def message_with_context(user_qusetion):

    # check retrived docs, print them undereach other
    if docs_list:
        docs_section = ""
        for i, doc in enumerate(docs_list, 1):
            docs_section += f"<doc {i}>\n{doc}\n</doc {i}>\n\n"
    else:
        docs_section = "No documents retrieved."


    asking_prompt =f""""""
    Depend on this user question:
    
    {user_qusetion}

    and the retrived legal doc:

    <docs>
    {docs_section}
    </docs>

    Answer guide:
    - read the doc well and use only the lines or laws related to user Question
    - if no doc and the user askes about previous chat answer him
    """"""

    message_history.append({"role": "user", "content": asking_prompt})

    chat_response = message_chat(message_history)

    message_history.append({"role": "assistant", "content": chat_response})


    return display(Markdown(chat_response))

"""

############################# Using LangChain ##############################

# Initialize LangChain Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2025-01-01-preview",
    temperature=0.2,
)

# Initialize LangChain Azure OpenAI LLM for reasoning
llm_reason = AzureChatOpenAI(
    azure_deployment="chatgpt-4o-latest",
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2025-01-01-preview",
    temperature=0.1,
)