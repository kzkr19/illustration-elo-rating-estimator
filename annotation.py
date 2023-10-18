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


def calculate_rating(compare_result: dict, files: List[str]):
    rating = {}

    for f in files:
        rating[f] = 1500

    for (f1, f2), result in compare_result.items():
        if result:
            update_rating(rating, f1, f2)
        else:
            update_rating(rating, f2, f1)

    return rating


def calculate_n_comparisons(compare_result: dict, files: List[str]):
    n_comparisons = {}

    for f in files:
        n_comparisons[f] = 0

    for f1, f2 in compare_result.keys():
        n_comparisons[f1] += 1
        n_comparisons[f2] += 1

    return n_comparisons


def select_file(files: List[str], n_comparisons: dict, compare_results: dict):
    candidate = sorted(files, key=lambda x: n_comparisons[x])

    for i in range(len(candidate)):
        for j in range(i + 1, len(candidate)):
            file1 = min(candidate[i], candidate[j])
            file2 = max(candidate[i], candidate[j])
            if (file1, file2) not in compare_results:
                return file1, file2

    raise ValueError('All pairs are compared.')


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
    # NOTE: compare_results[(file1, file2)] is True if file1 is winner
    compare_results = json.load(open(result_json_path, 'r'))\
        if result_json_path.exists() else {}

    # compare_count[file] is number of comparison of specified file
    n_comparisons = calculate_n_comparisons(compare_results, files)

    # evaluate
    for _ in range(n_rounds):
        file1, file2 = select_file(files, n_comparisons, compare_results)
        n_comparisons[file1] += 1
        n_comparisons[file2] += 1

        win, _lose = compare(file1, file2)
        compare_results[(file1, file2)] = win == file1
        json.dump(compare_results, open(result_json_path, 'w'))

    # calculate and save Elo rating
    # NOTE: rating[file] is elo rating
    rating = calculate_rating(compare_results, files)
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
