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


if __name__ == "__main__":
    make_dirs = str(input("Нужно создать иерархию папок для обучающих данных? (да/нет): "))
    if make_dirs == "да":
        dir_names = str(input("Введите список нужных папок по датам. Пример: 11.04.2025;14.01.2025\nВвод: ")).split(";")
        for name in dir_names:
            make_dirs_for_train(TRAIN_DATA_DIR + "\\" + name)
    print()
    input("Теперь перенесите все изображения и видео в нужные папки.\n"
          "В папки for_predictor нужно переместить заготовленные табличные файлы.\n"
          "Эти файлы должны быть названы 'predictor_labels.csv'\n"
          "Если вы все сделали, нажмите enter на клавиатуре: ")
    print()

    print("Теперь нужно предобработать изображения и видео.\n"
          "На каждой итерации нужно будет вводить:\n"
          "    название обрабатываемой папки, размер итоговых изображений и смещения при обработке\n"
          "Вот пример ввода: 11.04.2025;300;150,-23,20,0\n"
          "Последние 4 числа указывают как должны смещаться обрезаемые изображения по x и y для детектора и предиктора соответственно.\n"
          "То есть изображения для детектора сместятся на 150 пикселей вправо и на 23 пикселя вверх. (Начало координат левый верхний угол)")

    while True:
        inp = str(input("Введите данные для подготовки обучающих изображений (чтобы закончить, введите число 0): "))
        if inp == "0":
            break
        else:
            info = inp.split(";")
            info[2] = info[2].split(",")
            try:
                prepare_images([info[0]],
                               int(info[1]),
                               shift_list=[
                                   [
                                       [int(info[2][0]), int(info[2][1])], [int(info[2][2]), int(info[2][3])]
                                   ]
                               ])
                print("Данные предобработаны!\n"
                      "Если вам нужно поменять смещения, просто введите те же данные с другими смещениями.")
            except Exception as e:
                print(f"Что-то пошло не так: {e}")
    print()
    input("Теперь вам нужно сделать лейблы для каждого набора изображений.\n"
          "Можете посмотреть соответствующие инструкции по использованию label-studio.\n"
          "Если вы поместили файлы labels.json в нужные папки, можете переходить дальше.\n"
          "Чтобы продолжить, введите enter: ")
    print()
    print("Перейдем к обучению классификатора и предиктора.")
    image_size, train_names = str(input("Введите размер изображения и названия папок, изображения из которых вы хотите использовать для обучения.\n"
                            "Вводите в формате: 300;11.04.2025,17.01.2025\n"
                                        "Чтобы пропустить обучение моделей, введите 0;0\n"
                            "Введите данные: ")).split(";")
    if image_size != "0":
        try:
            train_classifier(train_names.split(","), train_data_dir=TRAIN_DATA_DIR, models_dir=MODELS_DIR)
            train_detector(train_names.split(","), train_data_dir=TRAIN_DATA_DIR, models_dir=MODELS_DIR, image_size=int(image_size))
            print("Модели обучены!")
        except Exception as e:
            print(f"Во время обучения произошла ошибка: {e}")
    print()
    train_names = str(input("Теперь подготовим изображения для обучения предиктора.\n"
                            "Введите список папок, которые хотите использовать. Пример: 11.04.2025;17.01.2025"
                            "Введите список: ")).split(";")
    print()
    try:
        classifier_model = newest(MODELS_DIR + "\\classifier")
        detector_model = newest(MODELS_DIR + "\\detector")
        prepare_predictor_images(classifier_name=classifier_model, detector_name=detector_model, dir_list=train_names)
        print("Данные были подготовлены для обучения предиктора")
    except Exception as e:
        print(f"Что-то пошло не так: {e}")

    try:
        train_predictor(train_names, TRAIN_DATA_DIR, MODELS_DIR)
        print("Предиктор был обучен!")
    except Exception as e:
        print(f"Ошибка во время обучения предиктора: {e}")

    print("Этап обучения и подготовки данных окончен. Дальнейшие действия по использованию модели проводятся вручную.")

# 11.04.2025;12.04.2025;13.04.2025;14.04.2025;17.01.2025;21.01.2025