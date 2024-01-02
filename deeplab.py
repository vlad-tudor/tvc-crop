import os
import torch
import urllib
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Set the TORCH_HOME environment variable (optional, for model caching)
os.environ['TORCH_HOME'] = 'models'

# Load the pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

# Check if GPU is available and move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the image preprocessing function
def preprocess_image(image_path):
    input_image = Image.open(image_path)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(input_image).unsqueeze(0)

# Process the image
image_path = './images/test2.jpeg'
input_batch = preprocess_image(image_path).to(device)

# Perform inference
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)

# Create a color palette and plot the result
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_batch.shape[2:])
r.putpalette(colors)

# Save the output
output_dir = './results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'output.png')
r.save(output_path)

print(f"Output saved to {output_path}")
