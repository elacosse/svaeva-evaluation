from typing import List

import openai
from svaeva_redux.langchain.redis import RedisChatMessageHistoryWindowed  # noqa
from svaeva_redux.schemas.redis import UserModel

from svaeva_evaluation.prompts.consonancia import introduction_audio_narrative


def retrieve_redis_windowed_chat_history_as_text(
    session_id: str, url: str, key_prefix: str, chat_history_length: int = 30, AI_NAME: str = "AI"
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
            text = f"{AI_NAME}: " + message.content
        conversation += text + "\n"
    return conversation


def construct_introduction_narrative(user: UserModel, conversation: str, save_to_user: bool = True) -> str:
    """Maps conversation into a narrative that will be played for room experience"""
    SYSTEM_PROMPT = introduction_audio_narrative

    # if they have a name they shared with us, we can use it in the prompt
    if user.first_name is not None:
        USER_PROMPT_TEMPLATE = f"Create a short introduction for {user.first_name} welcoming them by name to be read aloud based on the following conversation they had with an interviewer: \n\n{conversation}"
    else:
        USER_PROMPT_TEMPLATE = f"Create a short introduction (call them our unique visitor) to be read aloud based on the following conversation they had with an interviewer: \n\n{conversation}"
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(conversation=conversation)},
        ],
    )
    response_content = str(response.choices[0].message.content)
    if save_to_user:
        user.introduction_narrative = response_content
    return response_content


def conversation_summarizer(conversation: str) -> str:
    """Maps conversation into some string output"""

    SYSTEM_PROMPT = "I am a helpful text analyzer that knows how to summarize a text. I receive extra bonuses for accurate summaries that will make me filthy rich contigent on my performance."
    USER_PROMPT_TEMPLATE = f"Summarize the following conversation into bullet points removing everything that isn't involving the human:\n\n{conversation}"
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(conversation=conversation)},
        ],
    )
    response_content = str(response.choices[0].message.content)
    return response_content


def construct_word_narrative_without_hurt(conversation: str) -> List[str]:
    """Constructs a word narrative from a conversation"""

    SYSTEM_PROMPT = "You are a Sebaldian inspired novelists who is extrodinarily effective at storytelling focusing on immersion and emotional draw. Immersion is what transports readers into your story world. Emotional draw is what keeps them there. You are a master of both. You are inspired by the following conversation:\n\n{conversation}"
    USER_PROMPT_TEMPLATE = "Choose a series of 6 positively charged adjectives that are single words that directly reflect details within the conversation. These words should be uplifting and positive. Choose their order to reflect a narrative arc. Make sure the words are separated by a new line."
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(conversation=conversation)},
            {"role": "user", "content": USER_PROMPT_TEMPLATE},
        ],
    )
    response_content = str(response.choices[0].message.content)
    return response_content.split("\n")


def construct_word_narrative_with_hurt(conversation: str) -> List[str]:
    """Constructs a word narrative from a conversation"""

    SYSTEM_PROMPT = "You are a Sebaldian inspired novelists who is extrodinarily effective at storytelling focusing on immersion and emotional draw. Immersion is what transports readers into your story world. Emotional draw is what keeps them there. You are a master of both. You are inspired by the following conversation:\n\n{conversation}"
    USER_PROMPT_TEMPLATE = "Choose a series of 6 negatively charged adjectives that are single words that directly reflect details within the conversation. These words should be sad and negative. Choose their order to reflect a narrative arc. Make sure the words are separated by a new line."
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(conversation=conversation)},
            {"role": "user", "content": USER_PROMPT_TEMPLATE},
        ],
    )
    response_content = str(response.choices[0].message.content)
    return response_content.split("\n")
