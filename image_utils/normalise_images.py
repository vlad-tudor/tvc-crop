# normalise_images.py

from PIL import Image
from torchvision import transforms
import torch

def preprocess_image(image_path, size=(256, 256)):
    """
    Preprocess the image: resize, convert to tensor, and normalize.
    """
    input_image = Image.open(image_path)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(input_image).unsqueeze(0), input_image.size

def normPRED(d):
    """
    Normalize the prediction tensor.
    """
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn