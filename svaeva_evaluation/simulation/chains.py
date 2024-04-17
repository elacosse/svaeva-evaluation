from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


def chat_chain(
    chat_message_history,
    system_prompt: str = "You are an unhelpful assistant.",
    chat=ChatOpenAI(model="gpt-4", temperature=0),
):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | chat

    response = chain.invoke(
        {
            "messages": chat_message_history.messages,
        }
    )
    return response


async def async_chat_chain(
    chat_message_history,
    system_prompt: str = "You are an unhelpful assistant.",
    chat=ChatOpenAI(model="gpt-4", temperature=0),
):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | chat

    response = await chain.ainvoke(
        {
            "messages": chat_message_history.messages,
        }
    )
    return response
