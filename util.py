import glob
import clip
import torch
from typing import List
import numpy as np
from PIL import Image


def get_all_image_paths(dir):
    extensions = ["png", "jpg", "jpeg"]
    result = []

    for ext in extensions:
        result += glob.glob(dir + f"/*.{ext}")

    return result


def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    return device, model, preprocess


def calculate_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return similarity


def encode_images(models, image_paths: List[str]):
    device, model, preprocess = models

    for i, file in enumerate(image_paths):
        image = preprocess(Image.open(file)).unsqueeze(0).to(device)
        with torch.no_grad():
            x = model.encode_image(image)

        if 'x_train' not in locals():
            x_train = np.zeros((len(image_paths), x.shape[1]))

        x_train[i] = x.cpu().numpy()

    return x_train
