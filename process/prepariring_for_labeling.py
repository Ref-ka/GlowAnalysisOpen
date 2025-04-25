from process.image_preprocessing import preprocess_images
from process.video_to_images import cut_videos
from process.image_resizing import crop_images
from process.data_info import calculate_mean_std

from paths import TRAIN_DATA_DIR


def prepare_images(dir_list: list[str], image_size: int, shift_list: list[list[list]]):
    dir_list = list(map(lambda dir_name: TRAIN_DATA_DIR + "\\" + dir_name, dir_list))
    for i, directory in enumerate(dir_list):

        # preprocess images for detector
        cut_videos(directory + "\\videos",
                   directory + "\\images\\for_detector\\unprepared",
                   40)
        preprocess_images(directory + "\\images\\for_detector\\unprepared",
                          directory + "\\images\\for_detector\\preprocessed_1080")
        crop_images(directory + "\\images\\for_detector\\preprocessed_1080",
                    image_size,
                    "c",
                    directory + "\\images\\for_detector\\prepared",
                    shift_list[i][0])

        # preprocess images for predictor
        preprocess_images(directory + "\\images\\for_predictor\\unprepared",
                          directory + "\\images\\for_predictor\\preprocessed_1080")
        crop_images(directory + "\\images\\for_predictor\\preprocessed_1080",
                    image_size,
                    "c",
                    directory + "\\images\\for_predictor\\prepared",
                    shift_list[i][1])

    print(calculate_mean_std(dir_list))


if __name__ == "__main__":
    prepare_images(["17.01.2025"], 300, [[[80, -140], [0, -45]]])
