import base64
from openai import OpenAI
import gradio as gr
import tempfile
import os
from PIL import Image

# Initialize Qwen client
client = OpenAI(
    api_key="enteryourvalidkey",  # Replace with your DashScope API key
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Function to encode PIL image to base64 string
def encode_image_to_base64(pil_img):
    """Encode a PIL image to base64 string."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            pil_img.save(tmpfile, format="JPEG")
            temp_path = tmpfile.name

        with open(temp_path, "rb") as img_file:
            encoded_str = base64.b64encode(img_file.read()).decode("utf-8")

        os.unlink(temp_path)
        return f"data:image/jpeg;base64,{encoded_str}"
    except Exception as e:
        print(f"[ERROR] Failed to encode image: {e}")
        return None

# Function to detect objects in the image using Qwen
def detect_objects_with_qwen(pil_img):
    if pil_img is None:
        return "No image captured."

    encoded_url = encode_image_to_base64(pil_img)
    if not encoded_url:
        return "Failed to process image."

    try:
        response = client.chat.completions.create(
            model="qwen-vl-max",  # or "qwen-omni"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Ignore previous context, explain what you see now and always explain in bahasa indonesia."},
                        {"type": "image_url", "image_url": {"url": encoded_url}}
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] API call failed: {e}"


# Gradio Interface
with gr.Blocks(title="📷 Real-Time Object Detection with Qwen") as demo:
    gr.Markdown("## 🚦 Real-Time Object Detection with Qwen-VL / Qwen Omni")
    gr.Markdown("👁️‍🗨️ This detects objects from your webcam feed using Qwen.")

    with gr.Row():
        webcam = gr.Image(sources="webcam", 
                          label="Live Webcam Feed", 
                          type="pil", 
                          streaming=True, 
                          width=320, 
                          height=240)
    
    output = gr.Textbox(label="📦 Detected Objects", lines=5)

    # Stream frames and analyze every 3 seconds, stream_every=3
    webcam.stream(fn=detect_objects_with_qwen, inputs=[webcam], outputs=output, stream_every=3)

if __name__ == "__main__":
    demo.launch()