import gradio as gr
import requests
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()
ENDPOINT_URL = "https://api.runpod.ai/v2/qkqui1t394hjws/runsync"
API_KEY = os.getenv("API_KEY")

def encode_image_to_base64(image):
  buffered = BytesIO()
  image.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return img_str


def critic_image(image):
  img_base64 = encode_image_to_base64(image)
  payload = {
      "input": {
          "max_new_tokens": 512,
          "category": "General Visual Analysis",
          "image": img_base64
      }
  }

  headers = {
      "Authorization": API_KEY,
      "Content-Type": "application/json"
  }
  

  response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
  result = response.json()

  return result['output']['result'].strip()


demo = gr.Interface(
  fn=critic_image,
  inputs=gr.Image(type="pil"),
  outputs="text",
  title="Gemmarte",
  description="Upload an image and get a visual analysis in text form from the Gemmarte model." 
)


if __name__ == "__main__":
  demo.launch()