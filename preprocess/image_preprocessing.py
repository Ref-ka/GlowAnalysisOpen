from PIL import Image, ImageOps
import os

from paths import IMAGES_DIR


def preprocess_image(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with Image.open(image_path) as img:
        original_width, original_height = img.size
        target_size = (1080, 1080)

        # Step 1: Crop the image if it is larger in any direction
        if original_width > target_size[0] or original_height > target_size[1]:
            left = max(0, (original_width - target_size[0]) // 2)
            top = max(0, (original_height - target_size[1]) // 2)
            right = min(original_width, left + target_size[0])
            bottom = min(original_height, top + target_size[1])
            img = img.crop((left, top, right, bottom))

        # Step 2: Pad the image if it is smaller in any direction
        new_width, new_height = img.size
        pad_left = (target_size[0] - new_width) // 2
        pad_top = (target_size[1] - new_height) // 2
        pad_right = target_size[0] - new_width - pad_left
        pad_bottom = target_size[1] - new_height - pad_top

        if new_width < target_size[0] or new_height < target_size[1]:
            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))

        output_path = os.path.join(output_dir, os.path.basename(image_path))
        img.save(output_path)
        print(f"Processed image saved to: {output_path}")


def preprocess_images(image_dir: str, output_dir: str):
    for image_name in os.listdir(image_dir):
        preprocess_image(image_dir + "\\" + image_name, output_dir)
    print("Images have been prepared_for_detector successfully!!!")


if __name__ == "__main__":
    preprocess_image(str(IMAGES_DIR) + "/17.01.25/unprepared/1.2 В 0.09 А.png", str(IMAGES_DIR) + "/17.01.25/prepared_for_detector")
