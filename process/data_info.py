import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from paths import TRAIN_DATA_DIR


def calculate_mean_std(image_dirs: list[str]):
    # Sums for each channel
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    num_pixels = 0
    images_count = 0

    for dir_name in image_dirs:
        for directory_type in ["for_detector", "for_predictor"]:
            directory_path = dir_name + f"\\images\\{directory_type}\\prepared"
            directory = os.listdir(directory_path)
            for image_path in directory:
                # Open image and transform it into tensor
                image = Image.open(directory_path + "\\" + image_path).convert("RGB")
                image_tensor = ToTensor()(image)  # Преобразуем в тензор (C, H, W)

                # Count sums and quad sums for each channel
                channel_sum += image_tensor.view(3, -1).mean(dim=1).numpy()
                channel_sum_squared += (image_tensor.view(3, -1) ** 2).mean(dim=1).numpy()

                # Scaling overall amount of pixels
                num_pixels += image_tensor.size(1) * image_tensor.size(2)

        images_count += len(directory)

    # Calc mean and std
    mean = channel_sum / images_count
    std = np.sqrt(channel_sum_squared / images_count - mean ** 2)

    return mean, std


if __name__ == "__main__":
    dir_list = ["14.04.2025"]
    dir_list = list(map(lambda dir_name: TRAIN_DATA_DIR + "\\" + dir_name, dir_list))
    print(calculate_mean_std(dir_list))
