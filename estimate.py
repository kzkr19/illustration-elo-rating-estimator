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
from util import get_all_image_paths


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

    rating = json.load(open(rating_json_path, 'r'))
    y_train = np.zeros(len(image_paths))

    for i, file in enumerate(image_paths):
        y_train[i] = rating[file]

    return x_train, y_train


def train_core(x_train, y_train):
    # NOTE: output layer is Linear
    # https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_model = MLP(in_channels=512, hidden_channels=[256, 1]).to(device)

    optimizer = torch.optim.Adam(target_model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # split train and test
    dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

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

    if Path(preprcessed_dataset_path).exists():
        with open(preprcessed_dataset_path, 'rb') as f:
            x_train, y_train = pickle.load(f)
    else:
        # preprocess
        image_directory = Path(image_directory)
        image_paths = glob.glob(str(image_directory / '*.jpg')) + \
            glob.glob(str(image_directory / '*.png'))

        print("loading clip model...")
        models = load_clip_model()
        print("start preprocessing...")
        x_train, y_train = preprocess_dataset(
            models, image_paths, rating_json_path)
        print("finish preprocessing...")
        with open(preprcessed_dataset_path, 'wb') as f:
            pickle.dump((x_train, y_train), f)

    # training
    trained_model = train_core(x_train, y_train)

    # save trained model
    torch.save(trained_model, trained_model_path)


def estimate(model_path: str, image_folder: str):
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model = torch.load(model_path, map_location=device)
    # load clip model
    _, clip_model, preprocess = load_clip_model()

    result = {}
    image_paths = get_all_image_paths(image_folder)

    for image_path in image_paths:
        # load image
        image = Image.open(image_path)

        # preprocess
        image = preprocess(image).unsqueeze(0).to(device)

        # estimate
        with torch.no_grad():
            x = clip_model.encode_image(image)
            y = trained_model(x)

        result[image_path] = y.cpu().numpy()[0][0]
        print(f"{image_path}: {result[image_path]}")

    # sort by rating
    sorted_ratings = sorted(result.items(), key=lambda x: x[1])
    for i, (file, r) in enumerate(sorted_ratings):
        print(f'{i + 1}: ({r}) {file} ')


if __name__ == "__main__":
    import fire
    fire.Fire({
        "train": train,
        "estimate": estimate
    })
