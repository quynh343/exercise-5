import os
import base64
from pydub import AudioSegment
import tempfile
from groq import Groq

# Gửi ảnh URL cho Groq Vision
def generate_image_caption(image_url, api_key):
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        temperature=1,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content.strip()


def speech_to_text(audio_path, api_key):
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sound = AudioSegment.from_file(audio_path)
    sound = sound.set_frame_rate(16000).set_channels(1)
    sound.export(temp_wav.name, format="wav")

    with open(temp_wav.name, "rb") as f:
        audio_data = f.read()
        audio_b64 = base64.b64encode(audio_data).decode()

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="whisper-large-v3-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio."},
                    {"type": "audio", "audio": {"data": audio_b64, "format": "wav"}},
                ],
            }
        ],
        temperature=1,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content.strip()
