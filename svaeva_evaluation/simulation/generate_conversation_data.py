import os
import random
from typing import List

import openai
from dotenv import find_dotenv, load_dotenv

SYSTEM_PROMPT_SIMULATION = "I am an expert at crafting excellent, engaging dialogue focused on phenomenological interviewing techniques. I am a machine deployed at an art exhibition that has people from all walks of life coming to visit."


def select_topic(previous_selections: List[str]) -> str:
    # previous_selections_string = "\n".join(previous_selections)
    # USER_PROMPT = f"""Select a single condition one of your patients might suffer from the condition list. The list below is a compilation of things you've already selected from. You should encourage yourself to select a different condition than the one in the list unless it's a quite common condition. This must be a single disease or condition from the given conditions below. Give your response as a single entity making sure it's a single disease or condition. Do not include anything extra in your response. Just a condition from the list below. That's it. One thing. Make sure it's from the conditions!

    # Example:
    # Depression
    # POTS
    # Chronic hip pain
    # Chronic migraine syndrome
    # Lupus

    # # Previous selections:
    # # {previous_selections_string}"""

    # USER_PROMPT = """Select a single condition one of your patients might suffer from the condition list below.

    # Conditions to select from:
    # Depression
    # POTS
    # Chronic hip pain
    # Chronic migraine syndrome
    # Lupus"""
    # client = openai.OpenAI()

    # import outlines
    # model = models.openai("gpt-4", system_prompt="You are a useful assistant")
    # output = model.generate_choice("Generate a disease.", choices=["Lupus", "Depression"])
    # print(output)

    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT_SIMULATION},
    #         {"role": "user", "content": USER_PROMPT},
    #     ],
    # )
    # response_content = response.choices[0].message.content
    choices = [
        "depression",
        "postural orthostatic tachycardia syndrome",
        "chronic hip pain",
        "chronic migraine syndrome",
        "disc hernia",
    ]
    response_content = random.choice(choices)

    print(f"Selected topic: {response_content}")
    return response_content


def simulate_conversation(topic: str) -> str:
    USER_PROMPT_TEMPLATE = "Craft an AI dialogue centered on empathetic conversation management with a person suffering from the following condition: {topic}. This bot would conduct a phenomenological interview wherein it is capable of understanding and discussing personal experiences of hurt or suffering. The model should be skilled in displaying empathy, active listening, understanding the nuances of pain, and providing emotional support. It should also adapt to the user's emotional changes throughout the conversation and respond with appropriate understanding and potential suggestions for healthy coping mechanisms when necessary. Remember, the objective is to facilitate intentional, respectful, and supportive conversations on difficult experiences and emotions. First ask an introductory question focusing on the current day not knowing what the actual condition is. The simultation should be realistic in the sense that it's an actual human sending you text messages via their phone so they may be distracted. It's typical there's grammar or some spelling errors as is the case when people usually message each other on their phones. Also, the human may not aware of what he or she is suffering from, so they might not reference the specific condition saying they have it. They only talk about their own experience living with it and what that's like from the perspective of their own subjective experience."
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SIMULATION},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(topic=topic)},
        ],
    )
    response_content = str(response.choices[0].message.content)
    return response_content


def save_conversations_to_file(
    file_dir: str,
    filename: str,
    conversation: str,
) -> None:
    os.makedirs(file_dir, exist_ok=True)
    filename = f"{filename}"
    file_path = os.path.join(file_dir, filename)
    with open(file_path, "w") as file:
        file.write(conversation)

    print(f"saved filename {filename}")


def main():
    file_dir = "/Users/eric/Library/CloudStorage/Dropbox/git/github/conversation_clustering/data/raw/v4"
    topic_selections = []
    for i in range(44, 100):
        topic = select_topic(topic_selections)
        topic_selections.append(topic)
        conversation = simulate_conversation(topic)
        filename = f"{i}_{topic}.txt"
        save_conversations_to_file(file_dir, filename, conversation)


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    # DATA_DIR = os.environ.get("DATA_DIR")
    main()
    # asyncio.run(main())
