# Path: svaeva_evaluation/audio/generate.py
import asyncio
import os
import re

import dotenv
from elevenlabs import save
from elevenlabs.client import AsyncElevenLabs

dotenv.load_dotenv()

client = AsyncElevenLabs(
    api_key=os.getenv("ELEVEN_API_KEY"),
    # httpx_client=httpx.AsyncClient(...),
)


async def async_generate(
    text: str,
    voice_id: str,
    stability: float = 0.71,
    similarity_boost: float = 0.5,
    style: float = 0.0,
    use_speaker_boost: bool = True,
    save_path: str = None,
) -> None:
    # models = await eleven.models.get_all()
    # print(models)
    # response = await client.voices.get_all()
    # Voice(
    #         voice_id=voice_id,
    #         settings=VoiceSettings(
    #             stability=stability, similarity_boost=similarity_boost, style=style, use_speaker_boost=use_speaker_boost
    #         ),
    #     ),
    results = await client.generate(
        text=text,
        voice=voice_id,
        model="eleven_multilingual_v2",
    )
    out = b""
    async for value in results:
        out += value

    if save_path:
        save(out, save_path)


async def async_generate_audio_from_list(texts: list, voice_id: str, filename_prefix: str = "output"):
    tasks = []
    # get root path
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for i, text in enumerate(texts):
        filename = f"{filename_prefix}_{str(i)}.mp3"
        save_path = os.path.join(root_path, "data/audio", filename)
        task = async_generate(text, voice_id, save_path=save_path)
        tasks.append(task)
        if i % 5 == 0:  # Limit the number of concurrent tasks (hard limit for ElevenLabs API)
            await asyncio.gather(*tasks)
            tasks = []


def patternize_list(input_list):
    pattern = re.compile(r"[\d\.]+")
    filtered_list = [pattern.sub("", word) for word in input_list]
    pattern_list = [" ... ... ...".join([word] * 3) for word in filtered_list]
    return pattern_list


if __name__ == "__main__":
    # Test the function with the given list
    input_list = ["1. Chronic", "2. Struggling", "3. Disheartening", "4. Restricted", "5. Exhausting", "6. Painful"]
    result = patternize_list(input_list)
    print(result)
    asyncio.run(async_generate("Hello there!", voice_id="Rachel", save_path="output.mp3"))
