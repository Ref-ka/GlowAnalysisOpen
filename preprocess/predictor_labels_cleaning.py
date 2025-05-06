import os
import re

from paths import TRAIN_DATA_DIR


def rename_files(folder):
    """
    Функция для переименования файлов для обучения предиктора.
    Формат обработанных наименований: float(space)float.png Также может добавляться индекс _num перед .png, если есть одинаковые имена файлов
    :param folder: Путь к директории с изображениями для изменения имен
    :return:
    """
    number_pattern = re.compile(r'(\d+(?:\.\d+)?)')
    for filename in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, filename)):
            numbers = number_pattern.findall(filename)
            if len(numbers) >= 2:
                base_name = f"{numbers[0]} {numbers[1]}"
                new_name = f"{base_name}.png"
                old_path = os.path.join(folder, filename)
                new_path = os.path.join(folder, new_name)
                counter = 1
                # Add a counter if file exists
                while os.path.exists(new_path):
                    new_name = f"{base_name}_{counter}.png"
                    new_path = os.path.join(folder, new_name)
                    counter += 1
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_name}")
            else:
                print(f"Skipped (not enough numbers): {filename}")


# Используется автоматически в prepare_images
if __name__ == "__main__":
    rename_files(TRAIN_DATA_DIR + r"\11.04.2025\images\for_predictor\prepared")
