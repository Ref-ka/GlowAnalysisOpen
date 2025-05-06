from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

from labels_load import get_coordinates
from paths import TRAIN_DATA_DIR


def save_images_with_boxes(image_paths: list[str], boxes_list: list, save_dir: str):
    """
    Функция для визуализации размеченных данных, используется для проверки адекватности обучающих данных
    :param image_paths: Пути к изображениям
    :param boxes_list: Список границ для каждого изображения
    :param save_dir: Путь к директории для сохранения изображений с границами
    :return:
    """
    for i, image_path in enumerate(image_paths):
        # Открываем изображение
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Получаем список прямоугольников для текущего изображения
        boxes = boxes_list[i]

        # Рисуем прямоугольники
        draw.rectangle([tuple(boxes[:2]), tuple(boxes[2:])], outline="red", width=1)

        # Отображаем изображение
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.suptitle(os.path.basename(image_path))
        plt.axis("off")
        plt.title(f"Image {i+1}")
        plt.savefig(os.path.join(save_dir, os.path.basename(image_path)))
        plt.close()


# Пример использования
if __name__ == "__main__":
    dir_paths = ["14.04.2025", "17.01.2025", "11.04.2025"]
    for dir_path in dir_paths:
        for dir_type in ["predictor"]:  # "detector",
            directory = TRAIN_DATA_DIR + "\\" + dir_path + f"\\images\\for_{dir_type}"
            if "visualised_boxes" not in os.listdir(directory):
                os.mkdir(directory + "\\visualised_boxes")
            paths = os.listdir(directory + "\\prepared")

            image_paths = list(map(lambda x: directory + "\\prepared" + "\\" + x, paths))

            # Список координат прямоугольников для каждого изображения
            boxes_list = get_coordinates(directory + "\\labels.json", 300)

            # Визуализация
            save_images_with_boxes(image_paths, boxes_list, directory + "\\visualised_boxes")
