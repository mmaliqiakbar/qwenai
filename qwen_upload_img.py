import base64
from openai import OpenAI
import gradio as gr
import os

# Initialize client
client = OpenAI(
    api_key="enteryourvalidkey", # Replace with your DashScope API key
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Function to encode local image to base64 string with MIME type
def encode_image_to_base64(image_path):
    """Encode a local image to base64 string with proper MIME type."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_str = base64.b64encode(image_file.read()).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()[1:]  # Get extension without dot
        return f"data:image/{ext};base64,{encoded_str}"
    except Exception as e:
        print(f"[ERROR] Failed to encode {image_path}: {e}")
        return None

# Function to analyze an image using Qwen-VL model
def analyze_image_with_qwen(image, user_question):
    if image is None:
        return "Please upload an image."
    
    encoded_url = encode_image_to_base64(image)
    if not encoded_url:
        return "Failed to process image."

    try:
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_question},
                        {"type": "image_url", "image_url": {"url": encoded_url}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] API call failed: {e}"

# Gradio Interface
def web_analyze(image, question):
    return analyze_image_with_qwen(image, question)

interface = gr.Interface(
    fn=web_analyze,
    inputs=[
        gr.Image(label="Upload an Image", type="filepath"),  # Important: use 'file' type
        gr.Textbox(label="Your Prompt", value="Get information from the image and list all the parameter that you find in the image.")
    ],
    outputs=gr.Textbox(label="Qwen-VL Response"),
    title="Qwen-VL Image Analyzer (with Gradio)",
    description="Upload an image and ask any question about it.",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch()