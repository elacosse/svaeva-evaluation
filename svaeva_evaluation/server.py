import logging
import os

import dotenv
import redis
from apscheduler.schedulers.asyncio import BackgroundScheduler
from fastapi import FastAPI
from rich.logging import RichHandler

# from apscheduler.schedulers.background import BackgroundScheduler

FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("evaluation - server")
logging.getLogger("apscheduler").setLevel(logging.WARNING)
dotenv.load_dotenv(dotenv.find_dotenv())

# LANGSERVE_URL = "http://" + os.getenv("LANGSERVE_HOST") + ":" + os.getenv("LANGSERVE_PORT")
GROUP_ID = "metamersion3-test0"
PLATFORM_ID = "telegram"
CONVERSATION_ID = "consonancia"
KEY_PREFIX = f"{PLATFORM_ID}_{CONVERSATION_ID}:"


app = FastAPI(
    title="ConsonanciaIA Scheduler - Visualizer",
    version="1.0",
    description="SSS",
)

# Connect to Redis
redis_connection = redis.Redis(
    host=os.getenv("REDIS_HOST"), port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB_INDEX")
)


def render_graph(platform_id: str, group_id: str, conversation_id: str, interaction_count: int):
    """asdasdasd"""
    return 0


@app.on_event("startup")
def start_scheduler():
    scheduler = BackgroundScheduler()
    # scheduler = AsyncIOScheduler()
    scheduler.add_job(
        render_graph,
        "interval",
        seconds=5,
        args=[
            platform_id,
            group_id,
            conversation_id,
            interaction_count,
        ],  # Gifting at interactions in list
        max_instances=1,
    )
    # Update the network every 1 minute

    # scheduler.add_job(
    #     async_conversation_image_gifting_condition_interactions,
    #     "interval",
    #     minutes=0.05,  # 0.05 minutes = 3 seconds
    #     args=[GROUP_ID],  # Gifting at interactions 1 and 6.
    # )
    scheduler.start()


@app.get("/")
async def root():
    dotenv.load_dotenv()
    platform_id = os.getenv("PLATFORM_ID")
    group_id = os.getenv("GROUP_ID")
    conversation_id = os.getenv("CONVERSATION_ID")
    interaction_count = int(os.getenv("INTERACTION_COUNT", 2))
    key_prefix = f"{platform_id}_{conversation_id}:"

    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Root path: {root_path}")

    logger.info("Gathering Users from Redis...")
    logger.info("[red]Redis Host[/]: " + os.environ["REDIS_HOST"])
    logger.info("[red]Redis OM URL[/]: " + os.environ["REDIS_OM_URL"])
    logger.info(f"Group ID: {group_id}")
    logger.info(f"Key Prefix: {key_prefix}")
    logger.info(f"Interaction Count Filter: {interaction_count}")

    return {"message": "FastAPI server is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
