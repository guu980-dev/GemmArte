import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

visual_analysis = {
    "General Visual Analysis": "Create a detailed and cohesive analysis paragraph focusing exclusively on the visual characteristics, ensuring clarity and thorough examination.",
    "Form and Shape": "Provide a focused analysis that critically examines the form and shape of the object, highlighting its visual impact and structural elements.",
    "Symbolism and Iconography": "Explore the symbolism and iconography through an in-depth visual analysis, identifying significant symbols and their interpretative meanings.",
    "Composition": "Conduct a visual analysis that emphasizes the compositional elements, examining the arrangement and structural balance of the artwork.",
    "Light and Shadow": "Evaluate the effects of light and shadow through a detailed analysis, focusing on how these elements enhance the visual dynamics.",
    "Texture": "Conduct a visual analysis of texture, emphasizing the surface qualities and tactile illusions presented in the piece.",
    "Movement and Gesture": "Analyze the movement and gesture within the work, highlighting how these visual cues suggest motion and expression.",
    "Color Palette": "Examine the color palette through an exclusive visual analysis, focusing on color harmony and emotional tone.",
    "Line Quality": "Analyze the line quality, exploring the visual characteristics and expressiveness conveyed through line variation.",
    "Perspective": "Conduct a study of perspective, analyzing how depth and spatial relationships are visually represented.",
    "Scale and Proportion": "Evaluate the scale and proportion within the composition, analyzing how size relationships affect the visual coherence."
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "/paligemma-3b-pt-224"
model_path = "/GemmArte"

print("Loading model...")
processor = AutoProcessor.from_pretrained(model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_path)
print("Model loaded.")
print("Moving model to device...")
model.to(device)


def generate(image, category: str, max_new_tokens=512) -> str:
    prompt = visual_analysis.get(category)
    if not prompt:
        # Default to general visual analysis
        prompt = visual_analysis["General Visual Analysis"]

    inputs = processor(prompt, image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    return processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
