import glob
import torch
from torch.utils.data import random_split, TensorDataset, DataLoader
from torchvision.ops import MLP
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


def preprocess_images(models, image_paths: List[str]):
    device, model, preprocess = models

    for i, file in enumerate(image_paths):
        image = preprocess(Image.open(file)).unsqueeze(0).to(device)
        with torch.no_grad():
            x = model.encode_image(image)

        if 'x_train' not in locals():
            x_train = np.zeros((len(image_paths), x.shape[1]))

        x_train[i] = x.cpu().numpy()

    return x_train


def preprocess_dataset(models, image_paths: List[str], rating_json_path: str):
    x_train = preprocess_images(models, image_paths)

    rating = json.load(open(rating_json_path, 'r'))["data"]
    y_train = np.zeros(len(image_paths))
    for i, file in enumerate(image_paths):
        y_train[i] = rating[file]

    return x_train, y_train


def train_core(x_train, y_train, x_test, y_test):
    # NOTE: output layer is Linear
    # https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_model = MLP(in_channels=512, hidden_channels=[256, 1]).to(device)

    optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(x_train), torch.tensor(y_train))

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(
        torch.tensor(x_test), torch.tensor(y_test))

    for epoch in range(100):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = target_model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_pred = target_model(test_dataset[:][0])
            loss = loss_fn(y_pred, test_dataset[:][1])
            print(f'epoch: {epoch}, loss: {loss}')

    return target_model


def train(
        image_directory: str,
        rating_json_path: str,
        preprcessed_dataset_path: str,
        trained_model_path: str):

    if Path(preprcessed_dataset_path).exist():
        x_train, y_train = pickle.load(preprcessed_dataset_path)
    else:
        # preprocess
        image_paths = glob.glob(str(image_directory / '*.jpg')) + \
            glob.glob(str(image_directory / '*.png'))

        models = load_clip_model()
        x_train, y_train = preprocess_dataset(
            models, image_paths, rating_json_path)
        pickle.dump((x_train, y_train), preprcessed_dataset_path)

    # training
    x_train, y_train, x_test, y_test = random_split(
        x_train, y_train, [0.8, 0.2])
    trained_model = train_core(x_train, y_train, x_test, y_test)

    # save trained model
    torch.save(trained_model, trained_model_path)


if __name__ == "__main__":
    train()
