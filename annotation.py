import matplotlib.pyplot as plt
from pathlib import Path
import fire
import glob
import json
import random
import trueskill
from typing import List
from copy import deepcopy


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


def calculate_elo_rating(compare_result: dict, files: List[str]):
    rating = {}

    for f in files:
        rating[f] = 1500

    for (f1, f2), result in compare_result.items():
        if result:
            update_rating(rating, f1, f2)
        else:
            update_rating(rating, f2, f1)

    return rating


def calculate_trueskill_rating(compare_result: dict, files: List[str]):
    ratings = {f: trueskill.Rating() for f in files}

    for (f1, f2), result in compare_result.items():
        if result:
            ratings[f1], ratings[f2] = trueskill.rate_1vs1(
                ratings[f1], ratings[f2])
        else:
            ratings[f2], ratings[f1] = trueskill.rate_1vs1(
                ratings[f2], ratings[f1])

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

    for i in range(len(candidate)):
        for j in range(i + 1, len(candidate)):
            file1 = min(candidate[i], candidate[j])
            file2 = max(candidate[i], candidate[j])
            if (file1, file2) not in compare_results:
                return file1, file2

    raise ValueError('All pairs are compared.')


def annotate(
        image_directory: Path,
        comparison_result_path: Path,
        rating_json_path: Path,
        n_rounds: int = 100,
        show_result: bool = False,
        rating_type: str = 'trueskill'):
    """
    annotate rating with Elo rating method.

    Args:
        image_directory: Path to directory containing images.
        comparison_result_path: Json file path to save comparison results. If the file exists,
            the results will be appended.
        rating_json_path: Json file path to save rating.
        n_rounds: Number of rounds to annotate. 
        show_result: If True, show the result of annotation.
        rating_type: Rating type. 'elo' or 'trueskill'.
    """

    image_directory = Path(image_directory)
    comparison_result_path = Path(comparison_result_path)

    # load files
    files = glob.glob(str(image_directory / '*.jpg')) + \
        glob.glob(str(image_directory / '*.png'))
    # NOTE: list of [file1, file2, file1 == winner]
    compare_results_raw = json.load(open(comparison_result_path, 'r'))["data"]\
        if comparison_result_path.exists() else []
    compare_results = {(f1, f2): result for f1, f2,
                       result in compare_results_raw}

    # compare_count[file] is number of comparison of specified file
    n_comparisons = calculate_n_comparisons(compare_results, files)

    # evaluate
    for _ in range(n_rounds):
        file1, file2 = select_file(files, n_comparisons, compare_results)
        n_comparisons[file1] += 1
        n_comparisons[file2] += 1

        print("compare", file1, file2)
        win, _lose = compare(file1, file2)
        compare_results[(file1, file2)] = win == file1

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
    # TODO: add TrueSkill rating mode
    fire.Fire(annotate)


if __name__ == '__main__':
    main()
