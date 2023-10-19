import glob
import torch
import clip
import json
from PIL import Image
from typing import List
from pathlib import Path
import numpy as np
import pickle


def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    return device, model, preprocess


def encode_image(models, image_path):
    device, model, preprocess = models

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    return model.encode_image(image)


def preprocess_images(models, image_paths: List[str]):
    # TODO: impleent
    image_features = None
    return image_features


def preprocess_dataset(models, image_paths: List[str], rating_json_path: str):
    # TODO: implement
    return None, None, None, None


def train_core(x_train, y_train, x_test, y_test):
    from torchvision.ops import MLP

    # NOTE: output layer is Linear
    # https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    target_model = MLP(in_channels=512, hidden_channels=[256, 1])

    # TODO: implement

    return target_model


def train(
        image_directory: str,
        rating_json_path: str,
        preprcessed_dataset_path: str,
        trained_model_path: str):

    image_paths = glob.glob(str(image_directory / '*.jpg')) + \
        glob.glob(str(image_directory / '*.png'))
    rating = json.load(open(rating_json_path, 'r'))["data"]

    if Path(preprcessed_dataset_path).exist():
        x_train, y_train, x_test, y_test = \
            pickle.load(preprcessed_dataset_path)
    else:
        models = load_clip_model()
        x_train, y_train, x_test, y_test = \
            preprocess_dataset(models, image_paths, rating_json_path)
        pickle.dump((x_train, y_train, x_test, y_test),
                    preprcessed_dataset_path)

    trained_model = train_core(x_train, y_train, x_test, y_test)
    torch.save(trained_model, trained_model_path)


if __name__ == "__main__":
    train()
