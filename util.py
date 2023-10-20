import glob
import clip
import torch


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
