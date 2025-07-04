import json

from paths import TRAIN_DATA_DIR


def transform(value, image_size, constant=0):
    return round(((value + constant) / 100) * image_size, 3)


def normalize(value, image_size):
    return round(value / image_size, 3)


def get_coordinates(labels_file: str, image_size: int, normalization: bool = False):
    """
    Функция для подгрузки координат разметок из файла labels.json
    :param labels_file:
    :param image_size:
    :param normalization:
    :return:
    """
    with open(labels_file) as file:
        data = json.load(file)

    coordinates = []
    for image in data:
        x = image['annotations'][0]["result"][0]["value"]["x"]
        y = image['annotations'][0]["result"][0]["value"]["y"]
        w = image['annotations'][0]["result"][0]["value"]["width"]
        h = image['annotations'][0]["result"][0]["value"]["height"]

        # Без смещений!
        x1 = round((x / 100) * image_size, 3) - 10
        y1 = round((y / 100) * image_size, 3) - 10
        x2 = round(((x + w) / 100) * image_size, 3) + 10
        y2 = round(((y + h) / 100) * image_size, 3) + 10

        # Ограничение диапазона
        x1 = max(0, min(x1, image_size-1))
        y1 = max(0, min(y1, image_size-1))
        x2 = max(0, min(x2, image_size))
        y2 = max(0, min(y2, image_size))

        coordinates.append([x1, y1, x2, y2])

    if normalization:
        for line in coordinates:
            for i in range(len(line)):
                line[i] = round(line[i] / image_size, 3)
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
