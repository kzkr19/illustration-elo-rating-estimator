import matplotlib.pyplot as plt
from pathlib import Path
import fire
import glob
import json
import random
from typing import List


def compare(image1: str, image2: str):
    """
    Compare two images with GUI, and return the winner and loser. 
    """
    win = None
    lose = None

    def on_press(event):
        nonlocal win, lose
        if event.key == "left":
            win = image1
            lose = image2
            plt.close()
        elif event.key == "right":
            win = image2
            lose = image1
            plt.close()

    while win is None or lose is None:
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', on_press)

        plt.subplot(1, 2, 1)
        plt.imshow(plt.imread(image1))
        plt.subplot(1, 2, 2)
        plt.imshow(plt.imread(image2))
        plt.show()

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

    w_21 = 1 / (1 + 10 ** ((r1 - r2) / 400))

    ratings[winner] = r1 + k * w_21
    ratings[loser] = r2 - k * w_21


def calculate_rating(compare_result: dict):
    # TODO: implement
    rating = {}
    return rating


def calculate_n_comparison(compare_result: dict, files: List[str]):
    # TODO: implement
    compare_count = {}
    return compare_count


def annotate(
        image_directory: Path,
        result_json_path: Path,
        rate_json_path: Path,
        n_rounds: int = 100,
        show_result: bool = False):
    """
    annotate rating with Elo rating method.

    Args:
        image_directory: Path to directory containing images.
        result_json_path: Json file path to save results. If the file exists,
            the results will be appended.
        rate_json_path: Json file path to save rating.
        n_rounds: Number of rounds to annotate. 
        show_result: If True, show the result of annotation.
    """

    image_directory = Path(image_directory)
    result_json_path = Path(result_json_path)

    # load files
    files = glob.glob(str(image_directory / '*.jpg')) + \
        glob.glob(str(image_directory / '*.png'))
    # compare_results[(file1, file2)] s True if file1 is winner
    compare_results = json.load(open(result_json_path, 'r'))\
        if result_json_path.exists() else {}

    # compare_count[file] is number of comparison of specified file
    compare_count = calculate_n_comparison(compare_results, files)

    # evaluate
    for _ in range(n_rounds):
        # TODO: select file which has less comparison count
        file1 = random.choice(files)
        file2 = random.choice(files)
        while file1 == file2:
            file2 = random.choice(files)

        win, lose = compare(file1, file2)
        update_rating(compare_results, win, lose)
        json.dump(compare_results, open(result_json_path, 'w'))

    # calculate and save Elo rating
    rating = calculate_rating(compare_results)  # rating[file] is elo rating
    json.dump(rating, open(rate_json_path, 'w'))

    if show_result:
        # sort by rating
        sorted_ratings = sorted(rating.items(), key=lambda x: x[1])
        for i, (file, r) in enumerate(sorted_ratings):
            print(f'{i + 1}: ({r}) {file} ')


def main():
    fire.Fire(annotate)


if __name__ == '__main__':
    main()
