# in this code, we will use the OpenAI-compatible API for DashScope to 
# analyze local images using Qwen-VL model.

import os
import base64
from openai import OpenAI

# Configuration
IMAGE_FOLDER = r"C:\temp\plnima\img" # Change this to your local image folder
API_KEY = "enteryourvalidkey"  # Make sure this is set in environment

# this is the question we will ask the model
QUESTION = "Get information from the image and list all the parameter that you find in the image. "
VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

# Initialize the OpenAI-compatible client for DashScope
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Function to get list of valid image files in the specified folder
def get_image_files(folder_path):
    """Get list of valid image files in the folder."""
    try:
        return [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]
    except Exception as e:
        print(f"[ERROR] Unable to read folder: {e}")
        return []

# Function to encode image to base64 string with MIME type
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

# Function to analyze a single image using Qwen-VL model
def analyze_image_with_qwen(image_path):
    """Analyze a single image using Qwen-VL via OpenAI-compatible API."""
    encoded_url = encode_image_to_base64(image_path)
    if not encoded_url:
        return "[SKIPPED] Invalid or unreadable image."
    
    user_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUESTION},
                {"type": "image_url", "image_url": {"url": encoded_url}}
            ]
        }
    ]

    try:
        response = client.chat.completions.create(
            model="qwen-vl",  # You can also use 'qwen-vl-max'
            messages=user_prompt
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        return f"[ERROR] API call failed: {e}"

def main():
    print("🖼️ Analyzing images from local folder...\n")

    image_files = get_image_files(IMAGE_FOLDER)
    if not image_files:
        print(f"[INFO] No valid images found in {IMAGE_FOLDER}.")
        return

    # Process each image file and call function to analyze it
    for img_file in image_files:
        full_path = os.path.join(IMAGE_FOLDER, img_file)
        print(f"📂 Processing: {img_file}")
        result = analyze_image_with_qwen(full_path)
        print(f"📝 Result: {result}\n")

if __name__ == "__main__":
    main()