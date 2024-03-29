import os
import numpy as np
import torch
import cv2
from torchvision import transforms
from image_utils.normalise_images import preprocess_image
from PIL import Image

class DeepLabV3:
    def __init__(self):
        # Set the TORCH_HOME environment variable (optional, for model caching)
        os.environ['TORCH_HOME'] = 'models'

        # Load the pre-trained DeepLabV3 model
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
        self.model.eval()

        # Check if GPU is available and move the model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def infer(self, image_path):
        input_batch, original_size = preprocess_image(image_path)
        input_batch = input_batch.to(self.device)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # Add a dummy batch dimension, resize, then remove the dummy batch dimension
        output_predictions = output_predictions.unsqueeze(0)  # Add dummy batch dimension
        output_resized = transforms.functional.resize(output_predictions, original_size[::-1], interpolation=transforms.InterpolationMode.NEAREST)
        output_resized = output_resized.squeeze(0)  # Remove dummy batch dimension

        return output_resized

    def find_deeplab_bounding_box(self, deeplab_output):
        """
        Find the bounding box around all features found by DeepLabV3.
        """
        # Convert the output to a numpy array
        output_np = deeplab_output.byte().cpu().numpy()

        # Find contours and the bounding box
        contours, _ = cv2.findContours(output_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, 0, 0, 0  # No contours found

        x, y, w, h = cv2.boundingRect(contours[0])
        x_max, y_max = x + w, y + h

        for cnt in contours[1:]:
            x1, y1, w1, h1 = cv2.boundingRect(cnt)
            x_max = max(x_max, x1 + w1)
            y_max = max(y_max, y1 + h1)
            x, y = min(x, x1), min(y, y1)

        w, h = x_max - x, y_max - y
        return x, y, w, h

    def process_image(self, image_path):
        """
        Process an image using DeepLabV3 and return the bounding box.
        """
        deeplab_output = self.infer(image_path)
        return self.find_deeplab_bounding_box(deeplab_output)
    
    def process_image(self, image_path):
        deeplab_output = self.infer(image_path)
        bbox = self.find_deeplab_bounding_box(deeplab_output)
        return bbox
    
    def save_output(self, image_path, output_path):
        """
        Save the normalized output of the DeepLabV3 model.
        """
        deeplab_output = self.infer(image_path)

        # Normalize the output tensor to the range [0, 255]
        output_np = deeplab_output.cpu().numpy()
        output_np = output_np - output_np.min()  # Shift to start at zero
        if output_np.max() != 0:  # Avoid division by zero
            output_np = output_np / output_np.max() * 255
        else:
            output_np = np.zeros_like(output_np)  # Handle the case of a uniform array

        # Convert the numpy array to an image
        output_image = Image.fromarray(output_np.astype(np.uint8))

        # Save the image
        output_image.save(output_path)