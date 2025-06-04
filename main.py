import os
from dotenv import load_dotenv
import gradio as gr
from modules.groq_inference import multimodal_chat

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def gradio_multimodal_interface(text_input, image_url, audio_input):
    try:
        return multimodal_chat(text_input, image_url, audio_input, GROQ_API_KEY)
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Multimodal Chatbot using Groq")

    text_input = gr.Textbox(label="💬 Enter your message")
    image_url = gr.Textbox(label="🌐 Paste image URL (public)")
    audio_input = gr.Audio(type="filepath", label="🎤 Upload or record audio")

    btn = gr.Button("🚀 Send")
    output = gr.Textbox(label="📤 Chatbot Response")

    btn.click(
        fn=gradio_multimodal_interface,
        inputs=[text_input, image_url, audio_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
