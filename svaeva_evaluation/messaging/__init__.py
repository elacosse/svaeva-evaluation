# load environment variables
import json
import logging
import os
from datetime import datetime

import dotenv
import redis
from rich.logging import RichHandler

dotenv.load_dotenv()

RANDOM_SEED = os.getenv("RANDOM_SEED", 42)
# Enable logging
# logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("schedulers - server")
dotenv.load_dotenv(dotenv.find_dotenv())
# Connect to Redis
redis_connection = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    db=os.getenv("REDIS_DB_INDEX"),
)


def queue_message_to_user(user_id: str, text: str, queue_name: str) -> None:
    """ """
    try:
        # Create a dictionary with the base64 encoded image and any additional data
        data = {
            "text": text,
            "metadata": {
                "user_id": user_id,
                "timestamp": str(datetime.now()),
                "source": "svaeva_evaluation",
            },
        }
        # Serialize the dictionary to a JSON string
        data_json = json.dumps(data)

        # Put the JSON string on a Redis queue
        redis_connection.lpush("message_processing_queue", data_json)

        logger.info(f"Message queued for user: {user_id}")

    except Exception as e:
        logger.error(f"Failed to queue message for user {user_id}: {e}")
