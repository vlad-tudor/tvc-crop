import torch
import os
import cv2
import numpy as np
from .basnet_model import BASNet  # Adjust the import based on your file structure
from image_utils.normalise_images import preprocess_image, normPRED
from PIL import Image

class BASNetModel:
    def __init__(self, model_path):
        self.model = BASNet(3, 1)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

        # Check if GPU is available and move the model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate_saliency_map(self, image_path):
        """
        Generate a saliency map using the BASNet model.
        """
        input_tensor, original_size = preprocess_image(image_path)
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            d1, _, _, _, _, _, _, _ = self.model(input_tensor)
        pred = normPRED(d1[:, 0, :, :])
        return pred, original_size

    def find_saliency_bounding_box(self, saliency_map, original_size, tolerance=0):
        """
        Find the bounding box from the saliency map with an optional tolerance.
        """
        # Convert to numpy array and apply threshold
        saliency_np = (saliency_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
        saliency_np = cv2.threshold(saliency_np, 128, 255, cv2.THRESH_BINARY)[1]

        # Optionally apply blur
        saliency_np = cv2.GaussianBlur(saliency_np, (5, 5), 0)

        # Find contours and the bounding box
        contours, _ = cv2.findContours(saliency_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, 0, 0  # No contours found

        x, y, w, h = cv2.boundingRect(contours[0])
        for cnt in contours[1:]:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            x, y, w, h = min(x, x1), min(y, y1), max(x+w, x1+w1) - x, max(y+h, y1+h1) - y

        # Scale bounding box coordinates to match the original image size
        scale_x = original_size[0] / saliency_map.shape[-1]
        scale_y = original_size[1] / saliency_map.shape[-2]
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        # Apply tolerance
        x = max(x - tolerance, 0)
        y = max(y - tolerance, 0)
        w = min(w + 2 * tolerance, original_size[0] - x)
        h = min(h + 2 * tolerance, original_size[1] - y)

        return x, y, w, h

    def process_image(self, image_path, tolerance=0):
        """
        Process an image: generate a saliency map and compute the bounding box.
        """
        saliency_map, original_size = self.generate_saliency_map(image_path)
        bbox = self.find_saliency_bounding_box(saliency_map, original_size, tolerance)

        return bbox
    
    def save_output(self, image_path, output_path):
        """
        Save the output of the BASNet model processing.
        """
        saliency_map, _ = self.generate_saliency_map(image_path)
        saliency_output = (saliency_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
        saliency_image = Image.fromarray(saliency_output)
        saliency_image.save(output_path)