import matplotlib.pyplot as plt
from pathlib import Path
import fire
import glob
import json
import random
import trueskill
from typing import List
from copy import deepcopy
from util import get_all_image_paths
import enum
from typing import Tuple


class ComparisonResult:
    LEFT_WIN = 0
    RIGHT_WIN = 1
    DRAW = 2


def compare(image1: str, image2: str, figsize: Tuple[int, int]):
    """
    Compare two images with GUI, and return the winner and loser. 
    """
    result = None

    def on_press(event):
        nonlocal result
        if event.key == "left":
            result = ComparisonResult.LEFT_WIN
            plt.close()
        elif event.key == "right":
            result = ComparisonResult.RIGHT_WIN
            plt.close()
        elif event.key == "down":
            result = ComparisonResult.DRAW
            plt.close()

    while result is None:
        fig = plt.figure(figsize=figsize)
        fig.canvas.mpl_connect('key_press_event', on_press)

        plt.subplots_adjust(left=0.05, right=0.995, bottom=0.05, top=0.995)
        plt.subplot(1, 2, 1)
        im1 = plt.imread(image1)
        plt.imshow(im1)
        if len(im1.shape) == 2:
            plt.gray()
        plt.axis('off')

        plt.subplot(1, 2, 2)
        im2 = plt.imread(image2)
        plt.imshow(im2)
        if len(im2.shape) == 2:
            plt.gray()
        plt.axis('off')
        plt.show()

    return result


def update_rating(ratings: dict, left: str, right: str, result: ComparisonResult):
    default_rating = 1500
    k = 32

    # coefficient for draw
    dc1 = 1.0
    dc2 = 0.0

    if result == ComparisonResult.DRAW:
        dc1 = 0.5
        dc2 = 0.5

    winner = left if result == ComparisonResult.LEFT_WIN else right
    loser = right if result == ComparisonResult.LEFT_WIN else left

    if winner not in ratings:
        ratings[winner] = default_rating
    if loser not in ratings:
        ratings[loser] = default_rating

    r1 = ratings[winner]
    r2 = ratings[loser]

    w_21 = 1 / (1 + 10 ** ((r1 - r2) / 400))
    w_12 = 1 / (1 + 10 ** ((r2 - r1) / 400))

    # ratings[winner] = r1 + k * w_21
    # ratings[loser] = r2 - k * w_21
    ratings[winner] = r1 + dc1 * k * w_21 - dc2 * k * w_12
    ratings[loser] = r2 - dc1 * k * w_21 + dc2 * k * w_12


def calculate_elo_rating(compare_result: dict, files: List[str]):
    rating = {}

    for f in files:
        rating[f] = 1500

    for (f1, f2), result in compare_result.items():
        update_rating(rating, f1, f2, result)

    return rating


def calculate_trueskill_rating(compare_result: dict, files: List[str]):
    ratings = {f: trueskill.Rating() for f in files}

    for (f1, f2), result in compare_result.items():
        winner = f1 if result == ComparisonResult.LEFT_WIN else f2
        loser = f2 if result == ComparisonResult.LEFT_WIN else f1
        drawn = result == ComparisonResult.DRAW

        ratings[winner], ratings[loser] = trueskill.rate_1vs1(
            ratings[winner], ratings[loser], drawn=drawn)

    # NOTE: ignore uncertainty
    return {f: ratings[f].mu for f in files}


def calculate_n_comparisons(compare_result: dict, files: List[str]):
    n_comparisons = {}

    for f in files:
        n_comparisons[f] = 0

    for f1, f2 in compare_result.keys():
        n_comparisons[f1] += 1
        n_comparisons[f2] += 1

    return n_comparisons


def select_file(files: List[str], n_comparisons: dict, compare_results: dict):
    files = deepcopy(files)
    random.shuffle(files)
    candidate = sorted(files, key=lambda x: n_comparisons[x])

    # low n_comparisons comes first
    for i in range(len(candidate)):
        for j in range(i + 1, len(candidate)):
            file1 = min(candidate[i], candidate[j])
            file2 = max(candidate[i], candidate[j])
            if (file1, file2) not in compare_results:
                return file1, file2

    raise ValueError('All pairs are compared.')


def annotate(
        image_directory: Path,
        output_directory: Path,
        n_rounds: int = 100,
        show_result: bool = False,
        rating_type: str = 'trueskill',
        figsize=(10, 7)):
    """
    annotate rating with Elo rating method.

    Args:
        image_directory: Path to directory containing images.
        output_directory: Path to directory to save metadata.
        n_rounds: Number of rounds to annotate. 
        show_result: If True, show the result of annotation.
        rating_type: Rating type. 'elo' or 'trueskill'.
    """

    comparison_result_path = str(output_directory) + "/result.json"
    rating_json_path = str(output_directory) + "/rating.json"

    image_directory = Path(image_directory)
    comparison_result_path = Path(comparison_result_path)

    # load files
    files = get_all_image_paths(image_directory)
    # NOTE: list of [file1, file2, file1 == winner]
    compare_results_raw = json.load(open(comparison_result_path, 'r'))["data"]\
        if comparison_result_path.exists() else []
    compare_results = {(f1, f2): result for f1, f2,
                       result in compare_results_raw}

    # compare_count[file] is number of comparison of specified file
    n_comparisons = calculate_n_comparisons(compare_results, files)

    # evaluate
    for i in range(n_rounds):
        file1, file2 = select_file(files, n_comparisons, compare_results)
        n_comparisons[file1] += 1
        n_comparisons[file2] += 1

        print(f"{i} th compare", file1, file2)
        result = compare(file1, file2, figsize)
        compare_results[(file1, file2)] = result

        # save compare results
        compare_results_raw = {"data": [[f1, f2, result]
                               for (f1, f2), result in compare_results.items()]}
        json.dump(compare_results_raw, open(comparison_result_path, 'w'))

    # calculate and save rating
    # NOTE: rating[file] is rating
    if rating_type == 'elo':
        rating = calculate_elo_rating(compare_results, files)
    else:
        rating = calculate_trueskill_rating(compare_results, files)
    json.dump(rating, open(rating_json_path, 'w'))

    if show_result:
        # sort by rating
        sorted_ratings = sorted(rating.items(), key=lambda x: x[1])
        for i, (file, r) in enumerate(sorted_ratings):
            print(f'{i + 1}: ({r}) {file} ')


def main():
    fire.Fire(annotate)


if __name__ == '__main__':
    main()
