import json

from paths import TRAIN_DATA_DIR


def get_coordinates(labels_file: str, image_size: int):
    with open(labels_file) as file:
        data = json.load(file)

    coordinates = []
    for image in data:
        if image["drafts"]:
            coordinates.append(
                [
                    int((image['drafts'][0]["result"][0]["value"]["x"] / 100) * image_size),
                    int((image['drafts'][0]["result"][0]["value"]["y"] / 100) * image_size),
                    int(((image['drafts'][0]["result"][0]["value"]["x"] + image['drafts'][0]["result"][0]["value"][
                        "width"]) / 100) * image_size),
                    int(((image['drafts'][0]["result"][0]["value"]["y"]) + image['drafts'][0]["result"][0]["value"][
                        "height"]) / 100 * image_size)
                ]
            )
        else:
            coordinates.append(
                [
                    int((image['annotations'][0]["result"][0]["value"]["x"] / 100) * image_size),
                    int((image['annotations'][0]["result"][0]["value"]["y"] / 100) * image_size),
                    int(
                        ((image['annotations'][0]["result"][0]["value"]["x"] + image['annotations'][0]["result"][0]["value"][
                            "width"]) / 100) * image_size),
                    int(((image['annotations'][0]["result"][0]["value"]["y"]) +
                           image['annotations'][0]["result"][0]["value"][
                               "height"]) / 100 * image_size)
                ]
            )
    return coordinates


def get_glow_classes(label_file: str):
    with open(label_file) as file:
        data = json.load(file)

    glow_classes = []
    for image in data:
        if image["annotations"][0]["result"][0]["value"]["rectanglelabels"][0] == "no_glow":
            glow_classes.append(0)
        else:
            glow_classes.append(1)
    return glow_classes


if __name__ == "__main__":
    print(get_glow_classes(TRAIN_DATA_DIR + "\\14.04.2025\\images\\for_detector\\labels.json"))
