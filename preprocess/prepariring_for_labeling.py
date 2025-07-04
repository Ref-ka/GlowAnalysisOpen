from preprocess.image_preprocessing import preprocess_images
from preprocess.video_to_images import cut_videos
from preprocess.image_resizing import crop_images
# from predictor_labels_cleaning import rename_files

from paths import TRAIN_DATA_DIR


def prepare_images(dir_list: list[str],
                   image_size: int,
                   positions: list[str] = None,
                   shift_list: list[list[list]] = None
                   ):
    """
        Подготовка изображений к разметке.

        Args:
            dir_list (list[str]): Пути к директориям с данными.
            image_size (int): Размер подготовленных изображений.
            positions (list[str]): Позиция для обрезки изображений (left, right, top, bottom).
            shift_list (list[list[list]]): Сдвиг для обрезки видео (отдельные сдвиги для детектора и предиктора).
        """
    dir_list = list(map(lambda dir_name: TRAIN_DATA_DIR + "\\" + dir_name, dir_list))
    for i, directory in enumerate(dir_list):
        # preprocess images for detector
        cut_videos(directory + "\\videos",
                   directory + "\\images\\for_detector\\unprepared",
                   20)
        preprocess_images(directory + "\\images\\for_detector\\unprepared",
                          directory + "\\images\\for_detector\\preprocessed_1080")
        crop_images(directory + "\\images\\for_detector\\preprocessed_1080",
                    image_size,
                    directory + "\\images\\for_detector\\prepared",
                    shift_list[i][0] if shift_list else None,
                    positions[i] if positions else None)

        # preprocess images for predictor
        preprocess_images(directory + "\\images\\for_predictor\\unprepared",
                          directory + "\\images\\for_predictor\\preprocessed_1080")
        crop_images(directory + "\\images\\for_predictor\\preprocessed_1080",
                    image_size,
                    directory + "\\images\\for_predictor\\prepared",
                    shift_list[i][1] if shift_list else None,
                    position=(positions[i] if positions else None))
        # rename_files(directory + "\\images\\for_predictor\\prepared")


if __name__ == "__main__":
    # Пример использования
    prepare_images(["13.04.2025"], 300, shift_list=[[[-150, -150], [0, 0]]])
