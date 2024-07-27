import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image

# Load the pre-trained model and processor
model_name = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
pipe = pipe.to("cuda")  # Move to GPU if available

# Function to load an image from a local file
def load_image(file_path):
    return Image.open(file_path).convert("RGB")

# Path to the local image file
file_path = "vanGogh.PNG"

# Load the sketch image
sketch_image = load_image(file_path)

# Generate the realistic image from the sketch
prompt = "turn this sketch into a realistic image"
with torch.autocast("cuda"):
    output = pipe(prompt=prompt, image=sketch_image)

# Save and display the output image
output_image = output.images[0]
output_image.show()
output_image.save("output.png")
