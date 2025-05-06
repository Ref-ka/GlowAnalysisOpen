import json

from paths import TRAIN_DATA_DIR


def transform_coordinates(value, image_size, constant):
    return round(((value + constant) / 100) * image_size, 2)


def get_coordinates(labels_file: str, image_size: int):
    """
    Функция для подгрузки координат разметок из файла labels.json
    :param labels_file:
    :param image_size:
    :return:
    """
    with open(labels_file) as file:
        data = json.load(file)

    coordinates = []
    for image in data:
        coordinates.append(
            [
                transform_coordinates(image['annotations'][0]["result"][0]["value"]["x"], image_size, -1),
                transform_coordinates(image['annotations'][0]["result"][0]["value"]["y"], image_size, -3),
                transform_coordinates(
                    image['annotations'][0]["result"][0]["value"]["x"] + image['annotations'][0]["result"][0]["value"][
                        "width"],
                    image_size, 1
                ),
                transform_coordinates(
                    image['annotations'][0]["result"][0]["value"]["y"] + image['annotations'][0]["result"][0]["value"][
                        "height"],
                    image_size, 2
                )
            ]
        )
    return coordinates


def get_glow_classes(label_file: str):
    """
    Функция подгрузки лейблов для обучающих данных для предиктора
    :param label_file:
    :return:
    """
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
