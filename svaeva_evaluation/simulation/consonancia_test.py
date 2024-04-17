# simulate interactions between interviewer and interviewee

# save conversation history in redis database

# compute trajectories in embedding space

# cognitive maps in latent space

# cognitive alignment between interviewer and interviewee


# Bot disclaimer.


# Bot initial image

import re
from typing import Callable, TypedDict

from config import REDIS_URL
from langchain_community.chat_message_histories import (
    RedisChatMessageHistory,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_openai import ChatOpenAI
from svaeva_redux.langchain.redis import RedisChatMessageHistoryWindowed
from svaeva_redux.schemas.redis import ConversationModel

# if start the server for the first time, initialize the database
# redis_client = redis.StrictRedis(host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"))
# Get ConversationRedisModel according to the chain_type
models_list = ConversationModel.find((ConversationModel.chain_type == "chain_with_history")).all()
# Sort the models by date_created_timestamp and select.
models_list = sorted(models_list, key=lambda x: x.date_created_timestamp)
conversation_model = models_list[-1]
lm_system_prompt = conversation_model.lm_system_prompt
vlm_system_prompt = conversation_model.vlm_system_prompt

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", lm_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{human_input}"),
    ]
)
model_kwargs = {
    "top_p": conversation_model.top_p,
    "frequency_penalty": conversation_model.frequency_penalty,
    "presence_penalty": conversation_model.presence_penalty,
}
chain = prompt | ChatOpenAI(
    model_name=conversation_model.engine,
    temperature=conversation_model.temperature,
    max_tokens=conversation_model.max_tokens,
    model_kwargs=model_kwargs,
)


def _is_valid_identifier(value: str) -> bool:
    """Check if the value is a valid identifier."""
    # Use a regular expression to match the allowed characters
    valid_characters = re.compile(r"^[a-zA-Z0-9-_]+$")
    return bool(valid_characters.match(value))


def create_session_factory(
    chat_history_length: int,
) -> Callable[[str], BaseChatMessageHistory]:
    """Create a factory that can retrieve chat histories.

    The chat histories are keyed by user ID and conversation ID.§

    Args:
    user_id: str, conversation_id: str, platform_id: str

    Returns:
        A factory that can retrieve chat histories keyed by user ID and conversation ID.
    """

    def get_redis_chat_history(user_id: str, conversation_id: str, platform_id: str) -> RedisChatMessageHistory:
        """Get a chat history from a user id and conversation id."""
        if not _is_valid_identifier(user_id):
            raise ValueError(
                f"User ID {user_id} is not in a valid format. "
                "User ID must only contain alphanumeric characters, "
                "hyphens, and underscores."
                "Please include a valid cookie in the request headers called 'user-id'."
            )
        if not _is_valid_identifier(conversation_id):
            raise ValueError(
                f"Conversation ID {conversation_id} is not in a valid format. "
                "Conversation ID must only contain alphanumeric characters, "
                "hyphens, and underscores. Please provide a valid conversation id "
                "via config. For example, "
                "chain.invoke(.., {'configurable': {'conversation_id': '123'}})"
            )
        if not _is_valid_identifier(platform_id):
            raise ValueError(
                f"Platform ID {platform_id} is not in a valid format. "
                "Platform ID must only contain alphanumeric characters, "
                "hyphens, and underscores. Please provide a valid platform id "
                "via config. For example, "
                "chain.invoke(.., {'configurable': {'platform_id': '123'}})"
            )

        # session_id = f"{conversation_id}-{platform_id}"
        key_prefix = "message_store:"
        history = RedisChatMessageHistoryWindowed(
            session_id=user_id,
            url=REDIS_URL,
            key_prefix=key_prefix,
            chat_history_length=chat_history_length,
        )
        print(history)
        # history = _clip_to_valid_token_length(history)
        # history = _truncate_message_history(history, chat_history_length)
        return history

    return get_redis_chat_history


class InputChat(TypedDict):
    """Input for the chat endpoint."""

    human_input: str
    """Human input"""


chain_with_history = RunnableWithMessageHistory(
    chain,
    create_session_factory(conversation_model.chat_history_length),
    input_messages_key="human_input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="platform_id",
            annotation=str,
            name="Platform ID",
            description="Unique identifier for the platform.",
            default="",
            is_shared=True,
        ),
    ],
).with_types(input_type=InputChat)


def main():
    chain_with_history.invoke(
        {
            "human_input": "Hello, I am a bot. How can I help you today?",
            "configurable": {
                "user_id": "consonancia-test",
                "conversation_id": "1",
                "platform_id": "svaeva-evaluation",
            },
        }
    )


if __name__ == "__main__":
    main()
