import runpod
import base64
from io import BytesIO
from PIL import Image
from pipeline import generate


def decode_to_image_obj(base64_string):
    return Image.open(BytesIO(base64.b64decode(base64_string)))


def process_input(input):
    """
    Execute the application code
    """
    max_new_tokens = input['max_new_tokens']
    category = input['category']
    base64_string = input['image']

    image = decode_to_image_obj(base64_string)

    result = generate(decode_to_image_obj(image), category, max_new_tokens)
    result = "This is a placeholder result."

    return {
        "result": result
    }


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    """
    This is the handler function that will be called by RunPod serverless.
    """
    return process_input(event['input'])


if __name__ == '__main__':
    print("Starting RunPod serverless worker.")
    runpod.serverless.start({'handler': handler})
