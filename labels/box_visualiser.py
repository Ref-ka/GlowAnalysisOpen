from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

from labels_load import get_coordinates
from paths import TRAIN_DATA_DIR


def visualize_images_with_boxes(image_paths, boxes_list):
    for i, image_path in enumerate(image_paths):
        # Открываем изображение
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Получаем список прямоугольников для текущего изображения
        boxes = boxes_list[i]

        # Рисуем прямоугольники
        draw.rectangle([tuple(boxes[:2]), tuple(boxes[2:])], outline="red", width=3)

        # Отображаем изображение
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Image {i+1}")
        plt.show()


# Пример использования
if __name__ == "__main__":
    dir_path = TRAIN_DATA_DIR + "\\14.04.2025\\images\\for_predictor\\prepared"
    # Пути к изображениям
    paths = os.listdir(dir_path)
    image_paths = list(map(lambda x: dir_path + "\\" + x, paths))

    # Список координат прямоугольников для каждого изображения
    # Формат: [[left_x, upper_y, right_x, lower_y], ...]
    boxes_list = get_coordinates(TRAIN_DATA_DIR + "\\14.04.2025\\images\\for_predictor\\labels.json", 300)
    print(boxes_list)

    # Визуализация
    visualize_images_with_boxes(image_paths, boxes_list)
