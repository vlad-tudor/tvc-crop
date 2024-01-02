# image_process.py

from basnet.basnet import load_basnet_model, generate_saliency_map
from deeplab.deeplab import DeepLabV3  # Import the DeepLabV3 class
from image_utils.bounding_box import consolidate_bounding_boxes, find_saliency_bounding_box, find_deeplab_bounding_box
from PIL import Image
from torchvision import transforms

# Global variable to hold the loaded models
loaded_models = {}

def load_all_models():
    """
    Load all models and store them in the global 'loaded_models' dictionary.
    """
    # Load BASNet model
    basnet_model_path = 'models/basnet/basnet.pth'  # Adjust the path as needed
    loaded_models['basnet'] = load_basnet_model(basnet_model_path)

    # Load DeepLabV3 model
    loaded_models['deeplabv3'] = DeepLabV3()
    # print("Models loaded successfully.")


def process_single_image(image_path, tolerance=10):
    """
    Process a single image using the loaded models and return the cropped image.
    """
    # Retrieve models
    basnet_model = loaded_models.get('basnet')
    deeplab_model = loaded_models.get('deeplabv3')

    if not basnet_model or not deeplab_model:
        raise Exception("One or more models are not loaded.")

    # Generate saliency map using BASNet
    basnet_saliency, original_size = generate_saliency_map(image_path, basnet_model)
    basnet_saliency_resized = transforms.functional.resize(basnet_saliency, original_size[::-1])
    basnet_box = find_saliency_bounding_box(basnet_saliency_resized, tolerance)

    # Generate segmentation output using DeepLabV3
    deeplab_output = deeplab_model.infer(image_path)
    deeplab_box = find_deeplab_bounding_box(deeplab_output)

    # Consolidate bounding boxes from both models
    consolidated_box = consolidate_bounding_boxes([basnet_box])

    # Crop the original image based on the consolidated bounding box
    original_image = Image.open(image_path)
    cropped_image = original_image.crop(consolidated_box)

    return cropped_image