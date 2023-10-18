import matplotlib.pyplot as plt
from pathlib import Path
import fire
import glob
import json
import random


def compare(image1: str, image2: str):
    # TODO: implement
    win = image1
    lose = image2

    return win, lose


def update_rating(ratings: dict, winner: str, loser: str):
    default_rating = 1500
    k = 32

    if winner not in ratings:
        ratings[winner] = default_rating
    if loser not in ratings:
        ratings[loser] = default_rating

    r1 = ratings[winner]
    r2 = ratings[loser]

    e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))

    ratings[winner] = r1 + k * (1 - e1)
    ratings[loser] = r2 + k * (0 - e2)


def annotate(
        image_directory: Path,
        result_json_path: Path,
        mode: str = 'round robin',
        n_rounds: int = 100,
        show_result: bool = False):
    """
    annotate rating with Elo rating method.

    Args:
        image_directory: Path to directory containing images.
        result_json_path: Json file path to save results. If the file exists,
            the results will be appended.
        mode: Annotation mode. 'round robin' or 'random'.
        n_rounds: Number of rounds to annotate. Only used when mode is 'random'
        show_result: If True, show the result of annotation.
    """

    image_directory = Path(image_directory)
    result_json_path = Path(result_json_path)

    files = glob.glob(str(image_directory / '*.jpg')) + \
        glob.glob(str(image_directory / '*.png'))
    ratings = json.load(open(result_json_path, 'r'))\
        if result_json_path.exists() else {}

    if mode == 'round robin':
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                win, lose = compare(files[i], files[j])
                update_rating(ratings, win, lose)
    elif mode == 'random':
        for _ in range(n_rounds):
            file1 = random.choice(files)
            file2 = random.choice(files)
            while file1 != file2:
                file2 = random.choice(files)

            update_rating(ratings, win, lose)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    json.dump(ratings, open(result_json_path, 'w'))

    if show_result:
        # sort by rating
        sorted_ratings = sorted(ratings.items(), key=lambda x: x[1])
        for i, (file, rating) in enumerate(sorted_ratings):
            print(f'{i + 1}: ({rating}) {file} ')


def main():
    fire.Fire(annotate)


if __name__ == '__main__':
    main()
