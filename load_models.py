import os
import torch
from torchvision import models
import gdown

os.environ['TORCH_HOME'] = 'models'

def download_deeplabv3():
    # Load and cache the pre-trained DeepLabV3 model
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model = models.segmentation.deeplabv3_resnet50(pretrained=True, weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    model = models.resnet34(pretrained=True)
    print("DeepLabV3 model downloaded.")

def download_basnet():
    file_id = '1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu'
    basnet_dir = 'models/basnet'  # Directory where the model will be saved
    basnet_path = os.path.join(basnet_dir, 'basnet.pth')

    # Create the directory if it doesn't exist
    if not os.path.exists(basnet_dir):
        os.makedirs(basnet_dir)

    # Download the file
    gdown.download(url=f'https://drive.google.com/uc?id={file_id}', output=basnet_path, quiet=False)
    print("BASNet model downloaded.")


if __name__ == "__main__":
    download_deeplabv3()
    download_basnet()
