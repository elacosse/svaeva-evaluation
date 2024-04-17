# load environment variables
import os

import dotenv
import openai
from conversation_clustering.conversation import conversation_summarizer
from dotenv import find_dotenv, load_dotenv

dotenv.load_dotenv()

RANDOM_SEED = os.getenv("RANDOM_SEED", 42)


def load_text_file(file_path):
    try:
        with open(file_path, "r") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


def main():
    directory_path = "/Users/eric/Library/CloudStorage/Dropbox/git/github/conversation_clustering/data/raw/v4"
    files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]
    for file in files[0:2]:
        conversation = load_text_file(file)
        conversation_mapper_output = conversation_summarizer(conversation)
        print(conversation_mapper_output)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    main()
