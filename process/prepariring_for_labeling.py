from process.image_preprocessing import preprocess_images
from process.video_to_images import cut_videos
from process.image_resizing import crop_images

from paths import TRAIN_DATA_DIR


def prepare_for_detector(dir_list: list[str]):
    for directory in dir_list:
        data_dir = TRAIN_DATA_DIR + "\\" + directory

        # preprocess images for detector
        cut_videos(data_dir + "\\videos",
                   data_dir + "\\images\\for_detector\\unprepared",
                   40)
        preprocess_images(data_dir + "\\images\\for_detector\\unprepared",
                          data_dir + "\\images\\for_detector\\preprocessed_1080")
        crop_images(data_dir + "\\images\\for_detector\\preprocessed_1080",
                    300,
                    "c",
                    data_dir + "\\images\\for_detector\\prepared",
                    [-150, -150])

        # preprocess images for predictor
        preprocess_images(data_dir + "\\images\\for_predictor\\unprepared",
                          data_dir + "\\images\\for_predictor\\preprocessed_1080")
        crop_images(data_dir + "\\images\\for_predictor\\preprocessed_1080",
                    300,
                    "c",
                    data_dir + "\\images\\for_predictor\\prepared")


if __name__ == "__main__":
    prepare_for_detector(["14.04.2025"])
