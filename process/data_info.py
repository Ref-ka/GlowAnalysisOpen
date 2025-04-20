import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor


def calculate_mean_std(image_paths):
    # Sums for each channel
    channel_sum = np.zeros(3)
    channel_sum_squared = np.zeros(3)
    num_pixels = 0

    for image_path in image_paths:
        # Open image and transform it into tensor
        image = Image.open(image_path).convert("RGB")
        image_tensor = ToTensor()(image)  # Преобразуем в тензор (C, H, W)

        # Count sums and quad sums for each channel
        channel_sum += image_tensor.view(3, -1).mean(dim=1).numpy()
        channel_sum_squared += (image_tensor.view(3, -1) ** 2).mean(dim=1).numpy()

        # Scaling overall amount of pixels
        num_pixels += image_tensor.size(1) * image_tensor.size(2)

    # Calc mean and std
    mean = channel_sum / len(image_paths)
    std = np.sqrt(channel_sum_squared / len(image_paths) - mean ** 2)

    return mean, std


if __name__ == "__main__":
    image_dir = "detector_train"
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    mean, std = calculate_mean_std(image_paths)
    print(f"Mean: {mean}")
    print(f"Std: {std}")