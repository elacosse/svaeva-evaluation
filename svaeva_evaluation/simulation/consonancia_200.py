# Simulates 200 simultaneous users with infra.


import base64
import json
import logging
import os
from types import NoneType
from uuid import uuid4

import dotenv
import redis
from langchain.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langserve import RemoteRunnable
from redis_om.model.model import NotFoundError
from rich.logging import RichHandler

from svaeva_evaluation.simulation.utils import wait_time_model

dotenv.load_dotenv(dotenv.find_dotenv())
from svaeva_redux.schemas.redis import UserModel

# from svaeva_evaluation.simulation.chains import async_chat_chain

"""
Telegram Bot

Usage:
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""

NUM_USERS = 1
NUM_ITERATIONS = 1
DISCLAIMER_MESSAGE = """Disclaimer: Conversational AI is at the technological frontier and generate responses and outputs based on complex algorithms and machine learning techniques. Those responses or outputs may be inaccurate or indecent. Using this model you assume the risk of any harm caused by any response or output of the model. Your data will be processed according to our Privacy Policy. You may delete all your data by typing /delete. By using this model you agree to our Terms of Service. If you do not agree, please do not use this model."""


MAX_INTERACTIONS = 50
MAX_INTERACTION_MESSAGE = (
    "You have reached the maximum number of interactions for this session. Please try again tomorrow."
)
LANGSERVE_URL = "http://" + str(os.getenv("LANGSERVE_HOST")) + ":" + str(os.getenv("LANGSERVE_PORT"))
GROUP_ID = os.getenv("GROUP_ID")
PLATFORM_ID = os.getenv("PLATFORM_ID")
CONVERSATION_ID = os.getenv("CONVERSATION_ID")
FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("schedulers - server")
logger.info(f"Platform: {PLATFORM_ID} - Conversation: {CONVERSATION_ID} - Group: {GROUP_ID}")

redis_connection = redis.Redis(
    host=str(os.getenv("REDIS_HOST")),
    port=os.getenv("REDIS_PORT"),
    db=os.getenv("REDIS_DB_INDEX"),
)


class Message:
    def __init__(self):
        self.text = None
        self.effective_attachment = None

    def reply_text(self, message_content):
        print(message_content)

    def reply_html(self, message_content, reply_markup):
        print(message_content)


class Update:
    class effective_user:
        def __init__(self, id, first_name, username, language_code):
            self.id = id
            self.first_name = first_name
            self.username = username
            self.language_code = language_code

    def __init__(self, id, first_name, username, language_code):
        self.effective_user = self.effective_user(id, first_name, username, language_code)
        self.message = Message()


# Define a few command handlers. These usually take the two arguments update and context.
async def start(update: Update) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    logger.info(f"User {user.id} started the conversation.")


async def delete_command(update: Update) -> None:
    """Delete the user data."""
    UserModel.delete(update.effective_user.id)
    # conversation pk
    conversation_pk = f"{PLATFORM_ID}_{CONVERSATION_ID}:{update.effective_user.id}"
    redis_connection.delete(conversation_pk)
    # delete the user conversation history
    logger.info(f"User {update.effective_user.id} data have been deleted.")


async def call_user_update(update: Update) -> UserModel:
    """Get or create a user from the update."""
    try:
        user = UserModel.get(update.effective_user.id)
        user.interaction_count += 1
        user.save()
    except Exception as e:
        if isinstance(e, NotFoundError):
            logger.info(f"Creating new user: {str(update.effective_user.id)}")
            # print(PLATFORM_ID, CONVERSATION_ID, GROUP_ID)
            user = UserModel(
                id=str(update.effective_user.id),
                group_id=GROUP_ID,
                platform_id=PLATFORM_ID,
                first_name=update.effective_user.first_name,
                username=update.effective_user.username,
                language_code=update.effective_user.language_code,
                interaction_count=1,
            )
            user.save()

        else:
            logger.error(f"An error occurred: {e}")
    return user


def filter_text(text: str) -> str:
    if "'\"" in text:
        text = text.replace("'\"", "'")
    return text


async def process_message(update: Update) -> None:
    """Echo the user message."""
    logger.info(f"Processing message from user {update.effective_user.id}")
    chat_id = str(update.effective_user.id)
    # update.effective_user.is_bot
    user = await call_user_update(update)  # updates the interaction count!
    conversation_id = CONVERSATION_ID  # Key for message_store

    # Check if the user has reached the maximum number of interactions
    if user.interaction_count > MAX_INTERACTIONS:
        logger.info(f"User {chat_id} has reached the maximum number of interactions.")

    elif isinstance(update.message.effective_attachment, NoneType) and isinstance(update.message.text, str):
        human_input = update.message.text
        # Filter the text to remove any unwanted characters combos
        human_input = filter_text(human_input)

        chat = RemoteRunnable(LANGSERVE_URL, cookies={"user_id": chat_id})
        outgoing_message = chat.invoke(
            {"human_input": human_input},
            {
                "configurable": {
                    "conversation_id": conversation_id,
                    "platform_id": PLATFORM_ID,
                }
            },
        )

        if hasattr(outgoing_message, "content"):
            message_content = outgoing_message.content
        else:
            message_content = outgoing_message  # for the case of a str

        logger.info(f"Outgoing message: {message_content}")

    # elif isinstance(update.message.effective_attachment, telegram.Audio) or isinstance(
    #     update.message.effective_attachment, telegram.Voice
    # ):
    #     audio_file = await context.bot.get_file(update.message.effective_attachment)
    #     # filepath  = audio_file.file_path
    #     fill = io.BytesIO()
    #     await audio_file.download_to_memory(fill)
    #     # Encode the audio data in Base64
    #     fill.seek(0)  # reset the file pointer to the beginning
    #     encoded_audio_data = base64.b64encode(fill.read()).decode("utf-8")
    #     # Send the audio data to the langserve
    #     audio_stt = RemoteRunnable(LANGSERVE_URL + "/audio_stt")
    #     outgoing_message = audio_stt.invoke(
    #         {
    #             "audio": encoded_audio_data,
    #             "configurable": {
    #                 "user_id": chat_id,
    #                 "conversation_id": conversation_id,
    #                 "platform_id": PLATFORM_ID,
    #             },
    #         }
    #     )

    #     if hasattr(outgoing_message, "content"):
    #         message_content = outgoing_message.content
    #     else:
    #         message_content = outgoing_message
    #     await update.message.reply_text(message_content)

    # elif isinstance(update.message.effective_attachment, telegram.Document):
    #     print("NOT IMPLEMENTED: Document")
    #     # TODO: Implement document handling

    # elif isinstance(update.message.effective_attachment, tuple) and all(
    #     isinstance(item, telegram.PhotoSize) for item in update.message.effective_attachment
    # ):
    #     photo_file = await context.bot.get_file(update.message.effective_attachment[-1])
    #     fill = io.BytesIO()
    #     await photo_file.download_to_memory(fill)
    #     # Encode the photo data in Base64
    #     fill.seek(0)
    #     encoded_photo_data = base64.b64encode(fill.read()).decode("utf-8")

    #     # Send the photo data to the langserve
    #     image_vlm = RemoteRunnable(LANGSERVE_URL + "/image_vlm")
    #     outgoing_message = image_vlm.invoke(
    #         {
    #             "image": encoded_photo_data,
    #             "configurable": {
    #                 "user_id": chat_id,
    #                 "conversation_id": conversation_id,
    #                 "platform_id": PLATFORM_ID,
    #             },
    #         }
    #     )

    #     if hasattr(outgoing_message, "content"):
    #         message_content = outgoing_message.content
    #     else:
    #         message_content = outgoing_message
    #     await update.message.reply_text(message_content)


def remove_job_if_exists(name: str, context: str) -> bool:
    """Remove job with given name. Returns whether job was removed."""
    current_jobs = context.job_queue.get_jobs_by_name(name)
    if not current_jobs:
        return False
    for job in current_jobs:
        job.schedule_removal()
    return True


async def simulate_interaction(user_id: str, wait_time: float) -> None:
    """Simulate an interaction with the user."""

    update = Update(
        id=user_id,
        first_name="Test",
        username="test",
        language_code="en",
    )

    logger.info(f"Simulating interaction with user {update.effective_user.id} - waiting for {wait_time} seconds.")
    await asyncio.sleep(wait_time)
    chat_id = str(update.effective_user.id)
    conversation_id = CONVERSATION_ID
    # Create a random message for the user
    # human_message = generate_random_string(10)
    human_message = "Hi there! I'm so happy to see you."
    # # human_message = await async_chat_chain(chat_message_history, system_prompt=system_prompt, chat=llm)
    update.message.text = human_message
    await process_message(update)


async def callback_image_processing_queue(queue_name: str = "image_processing_queue"):
    # Collect everything on redis from cache
    try:
        while True:
            # Get the last JSON string from the Redis queue
            data_json = redis_connection.rpop(queue_name)
            if data_json is None:
                logger.info(f"Queue {queue_name} is empty from callback")
                return

            # Deserialize the JSON string to a dictionary
            data = json.loads(data_json)
            user_id = data["metadata"]["user_id"]
            caption = data["metadata"]["caption"]
            logger.info(f"Image dequeued from {user_id}")
            timestamp = data["metadata"]["timestamp"]

            image_bytes = base64.b64decode(data["image"])

            # download the image into a file to save
            image_path = f"/Users/eric/Library/CloudStorage/Dropbox/git/github/svaeva/svaeva_eric/svaeva-evaluation/data/images/{user_id}_{timestamp}.jpg"
            with open(image_path, "wb") as file:
                file.write(image_bytes)

            #
            # send to user
            # await context.bot.send_photo(
            #     chat_id=user_id, photo=image_bytes, caption=caption
            # )
            # await context.bot.send_message(
            #     chat_id=user_id, text=f"Image dequeued from {user_id}"
            # )

    except Exception as e:
        logger.error(f"Failed to dequeue and send image : {e}")


async def main() -> None:
    """Start the bot."""

    logger.info("Starting the simulation...")
    # Empty chat message history
    chat_message_history = ChatMessageHistory()
    # Create the chat model
    model = "gpt-4-turbo"
    temperature = 2.0
    max_tokens = 150
    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)

    # Generate a uuid for the user
    uuid_user = str(uuid4())

    user_ids = []
    # Create NUM_USERS users
    for i in range(NUM_USERS):
        user_id = f"consonancia-200-id-{str(i)}"
        update = Update(
            id=user_id,
            first_name="Test",
            username="test",
            language_code="en",
        )
        await delete_command(update)
        await start(update)
        await call_user_update(update)
        user_ids.append(user_id)
    logger.info(f"{NUM_USERS} users created: consonancia-200-id-*.")

    mean_wait_time = 10  # Mean wait time for the Poisson distribution
    std_wait_time = 5  # Standard deviation of the wait time for the Poisson distribution
    lower_bound = 0.5  # Lower bound of the truncation
    upper_bound = 30  # Upper bound of the truncation
    num_samples = NUM_USERS  # Number of samples to generate
    num_interactions_per_user = NUM_ITERATIONS  # Number of interactions per user

    for iter in range(num_interactions_per_user):
        # Generate truncated samples
        samples = wait_time_model(
            mean_wait_time=mean_wait_time,
            std_wait_time=std_wait_time,
            num_samples=num_samples,
            lb=lower_bound,
            ub=upper_bound,
        )

        tasks = []
        for i, wait_time in enumerate(samples):
            user_id = user_ids[i]
            task = simulate_interaction(user_id, wait_time)
            tasks.append(task)
            task = callback_image_processing_queue("image_processing_queue")
            tasks.append(task)
        # wait for all tasks to finish
        logger.info(f"Waiting for all interactions to finish - iteration {iter}")
        await asyncio.gather(*tasks)

        # Download images from the image queue into a folder

    logger.info("Simulation completed.")

    # job_queue = application.job_queue
    # job_image_processing_queue = job_queue.run_repeating(callback_image_processing_queue, interval=5, first=1)

    # Run the bot until the user presses Ctrl-C
    # application.run_polling(allowed_updates=Update.ALL_TYPES)


# async def start_scheduler():
#     # scheduler = BackgroundScheduler()
#     scheduler = AsyncIOScheduler()
#     scheduler.add_job(
#         async_conversation_image_gifting_condition_interactions,
#         "interval",
#         seconds=5,
#         args=[
#             GROUP_ID,
#             [1, 5, 13, 19, 27],
#             PLATFORM_ID,
#             KEY_PREFIX,
#         ],  # Gifting at interactions in list
#         max_instances=5,
#     )
#     # Get images from generation queue and place on processing queue for message sending
#     scheduler.add_job(
#         async_update_from_queue_onto_users_and_send_to_processing_queue,
#         "interval",
#         seconds=3,
#         args=["image_queue", "image_processing_queue"],
#     )

#     # Get images and feed them into video generation queue.
#     # scheduler.add_job(
#     #     async_update_from_queue_onto_users_and_send_to_processing_queue,
#     #     "interval",
#     #     seconds=10,
#     #     args=["video_queue", "video_processing_queue"],
#     # )

#     # scheduler.add_job(
#     #     async_conversation_image_gifting_condition_interactions,
#     #     "interval",
#     #     minutes=0.05,  # 0.05 minutes = 3 seconds
#     #     args=[GROUP_ID],  # Gifting at interactions 1 and 6.
#     # )
#     scheduler.start()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())


# import logging
# import os

# import dotenv
# import redis
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from fastapi import FastAPI
# from rich.logging import RichHandler

# # from apscheduler.schedulers.background import BackgroundScheduler
# from svaeva_schedulers.consonancia.gifter import (
#     async_conversation_image_gifting_condition_interactions,
# )

# FORMAT = "%(message)s"
# logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
# logger = logging.getLogger("svaeva-schedulers - server")
# logging.getLogger("apscheduler").setLevel(logging.WARNING)
# dotenv.load_dotenv(dotenv.find_dotenv())
# from svaeva_redux.schemas.utils import async_update_user_avatar

# # LANGSERVE_URL = "http://" + os.getenv("LANGSERVE_HOST") + ":" + os.getenv("LANGSERVE_PORT")
# GROUP_ID = "metamersion3-test0"
# PLATFORM_ID = "telegram"
# CONVERSATION_ID = "consonancia"
# KEY_PREFIX = f"{PLATFORM_ID}_{CONVERSATION_ID}:"
# # Enable logging
# # logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
# # # set higher logging level for httpx to avoid all GET and POST requests being logged
# # logging.getLogger("httpx").setLevel(logging.WARNING)
# # logger = logging.getLogger(__name__)


# app = FastAPI(
#     title="ConsonanciaIA Scheduler - Svaeva Redux",
#     version="1.0",
#     description="SSS",
# )

# # Connect to Redis
# redis_connection = redis.Redis(
#     host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB_INDEX")
# )


# # unqueue all images on queue
# async def async_update_from_queue_onto_users_and_send_to_processing_queue(
#     queue_name: str, processing_queue_name: str = "image_processing_queue"
# ) -> None:
#     """Dequeue images from the queue and update user avatars and embeddings.
#     Place the images on a processing queue for further processing, i.e., message gifting.

#     Args:
#         queue_name (str): The name of the queue to dequeue images from.
#         processing_queue_name (str, optional): The name of the queue to place the images on for further processing. Defaults to "image_processing_queue".

#     Returns:
#         None
#     """
#     try:
#         while True:
#             # Get the last JSON string from the Redis queue
#             data_json = redis_connection.rpop(queue_name)
#             if data_json is None:
#                 # logger.info(f"Queue {queue_name} is empty")
#                 return

#             # Deserialize the JSON string to a dictionary
#             data = json.loads(data_json)
#             user_id = data["metadata"]["user_id"]

#             # Update user avatar
#             await async_update_user_avatar(user_id, data["image"])

#             # place on separate queue for processing
#             redis_connection.lpush(processing_queue_name, data_json)
#             logger.info(f"Image dequeued from {queue_name} pushed on {processing_queue_name} for {user_id}")

#             # Decode the base64 encoded image
#             # image_bytes = base64.b64decode(data["image"])

#             # # Save the image to a file
#             # with open("image.jpg", "wb") as file:
#             #     file.write(image_bytes)
#     except Exception as e:
#         logger.error(f"Failed to dequeue and save image: {e}")


# @app.on_event("startup")
# @app.get("/")
# async def root():
#     return {"message": "FastAPI server is running"}


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app)
