from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image


def main():
  model_id = "google/paligemma-3b-pt-224"
  # model_path = "output/1727488022/checkpoint-112"
  model_path = "output/1727490265/checkpoint-450"
  model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
  processor = AutoProcessor.from_pretrained(model_id)

  # prompt = "Analyze image from a critic's point of view."
  prompt = "Please construct a formal analysis paragraph that is coherent and focuses solely on visual characteristic."
  image_file_path = "dataset/images/manual_test/starry_night.jpg"
  raw_image = Image.open(image_file_path)
  inputs = processor(prompt, raw_image, return_tensors="pt")
  output = model.generate(**inputs, max_new_tokens=20)

  # Starry Night
  print("Response: ", processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
  
 
if __name__ == "__main__":
  main()