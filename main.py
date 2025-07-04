import os
from utils.directory_handle import make_dirs_for_train
from preprocess.prepariring_for_labeling import prepare_images
from train.detector_train import train_detector
from train.classifier_train import train_classifier
from train.predictor_train import train_predictor
from paths import TRAIN_DATA_DIR, MODELS_DIR
from inference.image_processing import prepare_predictor_images


def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return str(os.path.basename(max(paths, key=os.path.getctime)))


def ask_yes_no(question):
    while True:
        ans = input(question + " (да/нет): ").strip().lower()
        if ans in ["да", "нет"]:
            return ans == "да"
        print("Пожалуйста, введите 'да' или 'нет'.")


def main():
    print("Добро пожаловать в систему подготовки и обучения моделей!")
    if ask_yes_no("Создать структуру папок для новых данных?"):
        dir_names = input("Введите даты через ; (например: 11.04.2025;14.01.2025): ").split(";")
        for name in dir_names:
            make_dirs_for_train(os.path.join(TRAIN_DATA_DIR, name.strip()))
        print("Папки созданы.\n")

    input("Перенесите изображения и видео в соответствующие папки. Нажмите Enter, когда будете готовы.")

    print("\n=== Предобработка изображений ===")
    while True:
        folder = input("Введите имя папки с данными (или 0 для завершения): ").strip()
        if folder == "0":
            break
        try:
            size = int(input("Введите размер итогового изображения (по умолчанию 300): ") or "300")
            shift = input("Введите смещения через , (например: 150,-23,20,0): ") or "150,-23,20,0"
            shift = [int(x) for x in shift.split(",")]
            n_images = int(input("Введите количество изображений, сделанных из видео (по умолчанию 10): "))
            prepare_images([folder], size, shift_list=[[[shift[0], shift[1]], [shift[2], shift[3]]]], n_images=n_images)
            print("Данные предобработаны!\n")
        except Exception as e:
            print(f"Ошибка: {e}")

    input("Сделайте разметку в label-studio и поместите labels.json в папки. Нажмите Enter для продолжения.")

    print("\n=== Обучение моделей ===")
    image_size = input("Введите размер изображения для обучения (по умолчанию 300): ") or "300"
    train_folders = input("Введите папки для обучения через , (например: 11.04.2025,17.01.2025): ").split(",")
    try:
        train_classifier(train_folders, train_data_dir=TRAIN_DATA_DIR, models_dir=MODELS_DIR)
        train_detector(train_folders, train_data_dir=TRAIN_DATA_DIR, models_dir=MODELS_DIR, image_size=int(image_size))
        print("Классификатор и детектор обучены!\n")
    except Exception as e:
        print(f"Ошибка при обучении: {e}")

    print("\n=== Подготовка и обучение предиктора ===")
    try:
        classifier_model = newest(os.path.join(MODELS_DIR, "classifier"))
        detector_model = newest(os.path.join(MODELS_DIR, "detector"))
        if ask_yes_no("Подготовить данные для предиктора?"):
            prepare_predictor_images(classifier_name=classifier_model, detector_name=detector_model, dir_list=train_folders)
        train_predictor(train_folders, TRAIN_DATA_DIR, MODELS_DIR)
        print("Предиктор обучен!\n")
    except Exception as e:
        print(f"Ошибка при обучении предиктора: {e}")

    print("Все этапы завершены! Модели готовы к использованию.")


if __name__ == "__main__":
    main()
