from PIL import Image, ImageOps
import os

from paths import IMAGES_DIR


def preprocess_image(image_path: str, output_dir: str):
    """
    Processes an image to ensure it is 1080x1080.
    If the image is larger, it is cropped. If smaller, it is padded with black.

    Parameters:
        image_path (str): Path to the input image.
        output_dir (str): Directory where the processed image will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    with Image.open(image_path) as img:
        original_width, original_height = img.size

        target_size = (1080, 1080)

        # Step 1: Crop the image if it is larger in any direction
        if original_width > target_size[0] or original_height > target_size[1]:
            # Calculate cropping box
            left = max(0, (original_width - target_size[0]) // 2)
            top = max(0, (original_height - target_size[1]) // 2)
            right = min(original_width, left + target_size[0])
            bottom = min(original_height, top + target_size[1])

            # Crop the image
            img = img.crop((left, top, right, bottom))

        # Step 2: Pad the image if it is smaller in any direction
        if img.size[0] < target_size[0] or img.size[1] < target_size[1]:
            img = ImageOps.pad(img, target_size, color=(0, 0, 0))

        # Save the processed image
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        img.save(output_path)
        print(f"Processed image saved to: {output_path}")


def preprocess_images(image_dir: list[str], output_dir: str):
    for image in image_dir:
        preprocess_image(image, output_dir)
    print("Images have been preprocessed successfully!!!")


if __name__ == "__main__":
    preprocess_image(str(IMAGES_DIR) + "/17.01.25/unprepared/1.2 В 0.09 А.png", str(IMAGES_DIR) + "/17.01.25/preprocessed")
