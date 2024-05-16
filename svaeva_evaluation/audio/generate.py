# Path: svaeva_evaluation/audio/generate.py
import asyncio
import os
import random
import re
from pathlib import Path

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
    save_path: Path,
    voice_id: str = "randomize",
    stability: float = 0.71,
    similarity_boost: float = 0.5,
    style: float = 0.0,
    use_speaker_boost: bool = True,
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

    if voice_id == "randomize":
        response = await client.voices.get_all()
        voice_id = random.choice(response.voices)

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


async def async_generate_audio_from_list(texts: list, voice_id: str, marker: str, save_dir: Path):
    tasks = []
    # get root path
    for i, text in enumerate(texts):
        save_path = save_dir / f"{i}_{marker}.mp3"
        task = async_generate(text, save_path, voice_id=voice_id)
        print(text)
        tasks.append(task)
        if i % 5 == 0:  # Limit the number of concurrent tasks (hard limit for ElevenLabs API)
            await asyncio.gather(*tasks)
            tasks = []


def patternize_list(input_list):
    pattern = re.compile(r"[\d\.]+")
    filtered_list = ['"' + pattern.sub("", word) + '"' for word in input_list]
    break_pause = [' <break time="2.0s" />', ' <break time="3.0s" />', ' <break time="4.0s" />']
    pattern_list = [random.choice(break_pause).join([word] * 3) for word in filtered_list]
    return pattern_list


if __name__ == "__main__":
    # Test the function with the given list
    input_list = ["1. Chronic", "2. Struggling", "3. Disheartening", "4. Restricted", "5. Exhausting", "6. Painful"]
    result = patternize_list(input_list)
    print(result)
    asyncio.run(async_generate("Hello there!", Path("output.mp3"), voice_id="Rachel"))
