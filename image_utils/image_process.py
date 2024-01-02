# image_process.py
import os
from basnet.basnet import BASNetModel
from deeplab.deeplab import DeepLabV3
from image_utils.bounding_box import consolidate_bounding_boxes
from PIL import Image
# Remove the import of find_saliency_bounding_box

# Global variable to hold the loaded models
loaded_models = {}

def load_all_models():
    """
    Load all models and store them in the global 'loaded_models' dictionary.
    """
    basnet_model_path = 'models/basnet/basnet.pth'  # Adjust the path as needed
    loaded_models['basnet'] = BASNetModel(basnet_model_path)

    # Load DeepLabV3 model
    loaded_models['deeplabv3'] = DeepLabV3()

def process_single_image(image_path, tolerance=10):
    """
    Process a single image using the loaded models and return the cropped image.
    """
    # Retrieve models
    basnet_model = loaded_models.get('basnet')
    deeplab_model = loaded_models.get('deeplabv3')
    if not basnet_model or not deeplab_model:
        raise Exception("One or more models are not loaded.")

    # Process the image using BASNet and DeepLabV3
    basnet_box = basnet_model.process_image(image_path, tolerance)
    deeplab_box = deeplab_model.process_image(image_path)

    # Consolidate bounding boxes from both models
    consolidated_box = consolidate_bounding_boxes([basnet_box, deeplab_box])

    # Crop the original image based on the consolidated bounding box
    original_image = Image.open(image_path)
    cropped_image = original_image.crop([consolidated_box[0], consolidated_box[1], consolidated_box[0] + consolidated_box[2], consolidated_box[1] + consolidated_box[3]])
   
    # Convert to RGB if necessary
    if cropped_image.mode == 'RGBA':
        cropped_image = cropped_image.convert('RGB')
        
    # Save and return paths to the processed images
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cropped_path = os.path.join('results', f'{base_name}_cropped.jpeg')
    basnet_path = os.path.join('results', f'{base_name}_basnet.jpeg')
    deeplab_path = os.path.join('results', f'{base_name}_deeplab.jpeg')


    cropped_image.save(cropped_path)

    # Assuming basnet_model and deeplab_model have methods to save their outputs
    basnet_model.save_output(image_path, basnet_path)
    deeplab_model.save_output(image_path, deeplab_path)


    return cropped_path, basnet_path, deeplab_path