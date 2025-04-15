from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os

from labels_load import get_coordinates


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
    # Пути к изображениям
    paths = os.listdir("new_cropped_images")
    paths.sort()
    image_paths = list(map(lambda x: "new_cropped_images/" + x, sorted(os.listdir("new_cropped_images")[:10])))

    # Список координат прямоугольников для каждого изображения
    # Формат: [[left_x, upper_y, right_x, lower_y], ...]
    boxes_list = get_coordinates("project-5-at-2025-03-30-14-06-3d3732a9.json")[:10]
    print(boxes_list)

    # Визуализация
    visualize_images_with_boxes(image_paths, boxes_list)
