import os

from paths import TRAIN_DATA_DIR


def make_dirs_for_train(directory: str):
    """
    Функция для создания нужного набора директорий в главной директории данных для обучения.
    :param directory:
    :return: None
    """
    for dir_type in ["for_detector", "for_predictor"]:
        for fin_dir_type in ["unprepared", "prepared", "preprocessed_1080"]:
            os.makedirs(directory + f"\\images\\{dir_type}\\{fin_dir_type}", exist_ok=True)
    os.mkdir(directory + "\\videos")


if __name__ == "__main__":
    make_dirs_for_train(TRAIN_DATA_DIR + "\\13.04.2025")
