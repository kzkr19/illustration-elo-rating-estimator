import glob
import torch
import clip
import json
from PIL import Image
from typing import List
from pathlib import Path


def load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)


    return device, model, preprocess

def encode_image(models, image_path):
    device, model, preprocess = models

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    return model.encode_image(image)

def calculate_image_features(image_paths: List[str]):
    # TODO: impleent
    image_features = None
    return image_features

def train(
        image_directory: str,
        rating_json_path: str,
        image_feature_path: str):
    
    image_paths = glob.glob(str(image_directory / '*.jpg')) + \
        glob.glob(str(image_directory / '*.png'))
    rating = json.load(open(rating_json_path, 'r'))["data"]
    if Path(image_feature_path).exist():
        # TODO: load image features
        image_features = None
    else:
        image_features = calculate_image_features(image_paths)
        # TODO: save image features
    
    # TODO: add flows



if __name__ == "__main__":
    train()
