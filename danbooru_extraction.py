import json
import os
import shutil
from IPython import embed
import sys
import random
import fire
import glob


def load_json(all_data, json_path):
    with open(json_path, 'r', encoding="utf-8") as f:
        while True:
            line = f.readline().strip()
            if line == '':
                break
            data = json.loads(line)
            all_data[int(data['id'])] = data


def load_all_json(json_dir):
    json_files = glob.glob(json_dir + "/*.json")
    all_data = {}

    for json_path in json_files:
        print("loading %s" % json_path)
        load_json(all_data, json_path)

    return all_data


def extract_id_from_path(image_path):
    return int(os.path.basename(image_path).split(".")[0])


def extract_high_scores(all_data, all_images, n_extract, sort_type):
    sort_rule = {
        "score": lambda x: x[1]["score"],
        "fav": lambda x: len(x[1]["favs"]),
    }
    score_list = [s for s in all_data.items()
                  if s[0] in all_images.keys()]
    score_list.sort(key=sort_rule[sort_type], reverse=True)

    extracted_image = score_list[:n_extract]

    return [d[0] for d in extracted_image]


def main(json_dir_or_path, image_dir, output_dir, n_extract=1000):
    # load metadata
    if os.path.isdir(json_dir_or_path):
        # load all json files
        # NOTE: this process need a lot of memory
        all_data = load_all_json(json_dir_or_path)
    else:
        all_data = {}
        load_json(all_data, json_dir_or_path)

    all_images = {
        extract_id_from_path(f): f for f in
        glob.glob(image_dir + "/*/*.jpg")
    }

    image_ids = list(all_images.keys())
    random.shuffle(image_ids)

    # extract list of image id which we need
    extracted_image_id = extract_high_scores(
        all_data, all_images, n_extract, "fav")

    print(extracted_image_id)
    # copy and save images and their extracted metadata
    subset_data = {}
    for image_id in extracted_image_id:
        image_path = all_images[image_id]
        output_path = os.path.join(output_dir, os.path.basename(image_path))

        print(output_path)
        shutil.copyfile(image_path, output_path)

        subset_data[image_id] = all_data[image_id]

    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding="utf-8") as f:
        f.write(json.dumps(subset_data))


if __name__ == '__main__':
    fire.Fire(main)
