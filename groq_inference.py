from modules.data_utils import generate_image_caption, speech_to_text
from groq import Groq

def query_groq_llm(prompt, api_key):
    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content.strip()

def multimodal_chat(user_text, image_url, audio_path, api_key):
    parts = []

    if image_url and image_url.strip().startswith("http"):
        caption = generate_image_caption(image_url, api_key)
        parts.append(f"🖼️ Image: {caption}")

    if audio_path:
        transcript = speech_to_text(audio_path, api_key)
        parts.append(f"🎤 Audio: {transcript}")

    if user_text.strip():
        parts.append(f"💬 User: {user_text}")

    full_prompt = "\n".join(parts) + "\n👉 Respond naturally:"
    return query_groq_llm(full_prompt, api_key)
