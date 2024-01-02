# basnet.py

import torch
from torchvision import transforms
from PIL import Image
from .basnet_model import BASNet  # Adjust the import based on your file structure

def load_basnet_model(model_path):
    """
    Load and return the BASNet model.
    """
    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net

def preprocess_image(image_path):
    """
    Preprocess the image: resize, convert to tensor, and normalize.
    """
    input_image = Image.open(image_path)
    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
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

def generate_saliency_map(image_path, model):
    """
    Generate a saliency map using the BASNet model.
    """
    input_tensor, original_size = preprocess_image(image_path)
    with torch.no_grad():
        d1, _, _, _, _, _, _, _ = model(input_tensor)
    pred = normPRED(d1[:, 0, :, :])
    return pred, original_size