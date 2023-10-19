import glob


def get_all_image_paths(dir):
    extensions = ["png", "jpg", "jpeg"]
    result = []

    for ext in extensions:
        result += glob.glob(dir + f"/*.{ext}")

    return result
