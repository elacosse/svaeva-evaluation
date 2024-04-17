import asyncio
import logging
import os

import dotenv
from langchain.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langserve import RemoteRunnable
from rich.logging import RichHandler
from svaeva_redux.langchain.redis import RedisChatMessageHistoryWindowed  # noqa

from svaeva_evaluation.prompts.consonancia import arthritis, breakup, chronic_hip_pain, disc_hernia, pots


def retrieve_redis_windowed_chat_history_as_text(
    session_id: str, url: str, key_prefix: str, chat_history_length: int = 30
) -> str:
    """
    Retrieve the chat history from Redis and return it as a string formatted for ingestion

    Args:
        session_id (str): The session id.
        url (str): The url of the Redis server.
        key_prefix (str): The key prefix for the chat history.
        chat_history_length (int): The length of the chat history.

    Returns:
        str: The chat history as a string.
    """

    history = RedisChatMessageHistoryWindowed(
        session_id=session_id,
        url=url,
        key_prefix=key_prefix,
        chat_history_length=chat_history_length,
    )

    conversation = ""
    for message in history.messages:
        if message.type == "human":
            text = "Human: " + message.content
        elif message.type == "ai":
            text = "AI: " + message.content
        conversation += text + "\n"
    return conversation


def compute_conversation_embedding_and_update_user(
    user_id: str,
    url: str,
    key_prefix: str,
    chat_history_length: int = 30,
    model: str = "text-embedding-3-large",
) -> None:
    """Update the conversation embedding for a user.

    Args:
        user_id (str): The user ID.

    Returns:
        None
    """

    client = OpenAI()

    try:
        text = retrieve_redis_windowed_chat_history_as_text(
            user_id, url, key_prefix, chat_history_length=chat_history_length
        )
        conversation_embedding = client.embeddings.create(input=[text], model=model).data[0].embedding
        update_user_conversation_embedding(user_id, conversation_embedding)
    except Exception as e:
        logger.error(f"Failed to update user conversation embedding: {e}")


FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("svaeva-evaluation - consonancia")


async def simulate_conversation(
    initial_interviewee_message: str,
    iterations: int,
    system_prompt: str,
    llm: ChatOpenAI,
    user_id: str,
    conversation_id: str,
    platform_id: str,
    group_id: str,
):
    chat_message_history = ChatMessageHistory()

    user = UserModel(id=user_id, group_id=group_id, platform_id=platform_id)
    user.save()
    chat = RemoteRunnable("http://0.0.0.0:8000/", cookies={"user_id": user_id})
    response = await chat.ainvoke(
        {"human_input": initial_interviewee_message},
        config={"configurable": {"user_id": user_id, "conversation_id": conversation_id, "platform_id": platform_id}},
    )
    chat_message_history.add_user_message(initial_interviewee_message)
    chat_message_history.add_ai_message(response["content"])
    # simulate conversation
    for i in range(iterations):
        # interviewee
        interviewee_message = await async_chat_chain(chat_message_history, system_prompt=system_prompt, chat=llm)
        interviewee_message = interviewee_message.content
        # logger.info("Human: " + interviewee_message)
        chat_message_history.add_user_message(interviewee_message)
        # interviewer
        response = await chat.ainvoke(
            {"human_input": interviewee_message},
            config={
                "configurable": {"user_id": user_id, "conversation_id": conversation_id, "platform_id": platform_id}
            },
        )
        chat_message_history.add_ai_message(response["content"])
        # logger.info("AI: " + response.content)
    # logger.info(chat_message_history)


async def main():
    platform_id = "svaeva-redux"
    group_id = "consonancia"
    conversation_id = "sim-consonancia"

    model = "gpt-4-0125-preview"
    temperature = 0.7
    max_tokens = 150

    initial_interviewee_message = "Hi"
    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    prompts = dict()
    prompts["disc_hernia"] = disc_hernia
    prompts["arthritis"] = arthritis
    prompts["chronic_hip_pain"] = chronic_hip_pain
    prompts["pots"] = pots
    prompts["breakup"] = breakup
    iterations = 6

    tasks = []
    for key, value in prompts.items():
        condition = key
        system_prompt = value
        for i in range(1, 7):
            user_id = f"sim-consonancia-{condition}-{str(i).zfill(3)}"
            logger.info(f"Starting conversation for user: {user_id}")
            task = simulate_conversation(
                initial_interviewee_message,
                iterations,
                system_prompt,
                llm,
                user_id,
                conversation_id,
                platform_id,
                group_id,
            )
            tasks.append(task)
    logger.info("Waiting for all conversations to finish")

    await asyncio.gather(*tasks)

    logger.info("All conversations finished")

    logger.info("Computing conversation embeddings and updating users")
    key_prefix = f"{platform_id}_{conversation_id}:"
    for key, value in prompts.items():
        condition = key
        for i in range(1, 7):
            user_id = f"sim-consonancia-{condition}-{str(i).zfill(3)}"
            logger.info(f"Computing conversation embedding for user: {user_id}")
            compute_conversation_embedding_and_update_user(
                user_id,
                os.environ["REDIS_OM_URL"],
                key_prefix,
                chat_history_length=30,
                model="text-embedding-3-large",
            )


if __name__ == "__main__":
    dotenv.load_dotenv()
    from openai import OpenAI
    from svaeva_redux.schemas.redis import UserModel
    from svaeva_redux.schemas.utils import (
        update_user_conversation_embedding,
    )

    from svaeva_evaluation.simulation.chains import async_chat_chain

    asyncio.run(main())
