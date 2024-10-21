import gradio as gr
import requests
import base64
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import deepl

load_dotenv()
ENDPOINT_URL = "https://api.runpod.ai/v2/qkqui1t394hjws/runsync"
INFERENCE_API_KEY = os.getenv("INFERENCE_API_KEY")
TRANSLATE_API_KEY = os.getenv("TRANSLATE_API_KEY")

def encode_image_to_base64(image):
  buffered = BytesIO()
  image.save(buffered, format="JPEG")
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  return img_str


def translate_en_to_ko(text):
  translator = deepl.Translator(TRANSLATE_API_KEY)
  try:
      result = translator.translate_text(text, target_lang="KO")
      return result.text
  except deepl.DeepLException as e:
      return f"번역 중 오류 발생: {str(e)}"


def critic_image(image, language, category):
  img_base64 = encode_image_to_base64(image)
  payload = {
      "input": {
          "max_new_tokens": 512,
          "category": category,
          "image": img_base64
      }
  }

  headers = {
      "Authorization": INFERENCE_API_KEY,
      "Content-Type": "application/json"
  }
  

  response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
  result = response.json()

  analysis_result = result['output']['result'].strip()

  if language == "KO":
      return translate_en_to_ko(analysis_result)  # 한국어로 번역 후 반환
  else:
      return analysis_result  # 영어 그대로 반환


categories = [
  'General Visual Analysis', 'Form and Shape', 'Symbolism and Iconography', 
  'Composition', 'Color Palette', 'Light and Shadow', 'Texture', 
  'Movement and Gesture', 'Line Quality', 'Perspective', 'Scale and Proportion'
]


demo = gr.Interface(
  fn=critic_image,
  inputs=[
    gr.Image(type="pil"),
    gr.Radio(choices=["EN", "KO"], label="Select Language", value="EN"),
    gr.Dropdown(choices=categories, label="Select Category", value="General Visual Analysis")
  ],
  outputs="text",
  title="Gemmarte",
  description="Upload an image and get a visual analysis in text form from the Gemmarte model." 
)


if __name__ == "__main__":
  demo.launch()