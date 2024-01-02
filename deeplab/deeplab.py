import os
import torch
from PIL import Image
from torchvision import transforms

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

    def preprocess_image(self, image_path):
        input_image = Image.open(image_path)
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(input_image).unsqueeze(0), input_image.size

    def infer(self, image_path):
        input_batch, original_size = self.preprocess_image(image_path)
        input_batch = input_batch.to(self.device)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # Resize output to match original image size
        output_resized = transforms.functional.resize(output_predictions, original_size[::-1], interpolation=transforms.InterpolationMode.NEAREST)

        return output_resized
