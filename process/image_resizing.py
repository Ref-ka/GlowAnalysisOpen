from PIL import Image
import os


def get_crop_box(img_w: int, img_h: int, crop_size: int, pos: str, shift) -> tuple[float, float, float, float]:
    if shift:
        shift_x, shift_y = shift[0], shift[1]
    else:
        shift_x, shift_y = 0, 0
    # Default to center
    left = (img_w - crop_size) // 2 + shift_x
    top = (img_h - crop_size) // 2 + shift_y

    if pos == "l":  # left center
        left = 0
    elif pos == "r":  # right center
        left = img_w - crop_size
    elif pos == "t":  # top center
        top = 0
    elif pos == "b":  # bottom center
        top = img_h - crop_size
    elif pos == "tl":  # top-left
        left = 0
        top = 0
    elif pos == "tr":  # top-right
        left = img_w - crop_size
        top = 0
    elif pos == "lb":  # left bottom
        left = 0
        top = img_h - crop_size
    elif pos == "rb":  # right bottom
        left = img_w - crop_size
        top = img_h - crop_size

    return left, top, left + crop_size, top + crop_size


def crop_images(data_path: str, crop_size: int, position: str, save_dir: str, shift=None):
    """
    Crops square images from given paths to the specified size and position.

    Args:
        data_path (str): Path of data folder.
        crop_size (int): Size of the square crop.
        position (str): Crop position ("l", "lu", "c", "b", etc.).
        save_dir (str): Directory to save cropped images.
        shift (list[int] or None): shift for images by X and Y axes
    """
    image_paths = os.listdir(data_path)

    for img_path in image_paths:
        with Image.open(data_path + "\\" + img_path) as img:
            img_w, img_h = img.size
            if crop_size > img_w or crop_size > img_h:
                raise ValueError(f"Crop size {crop_size} is larger than image size {img.size} for {img_path}")

            box = get_crop_box(img_w, img_h, crop_size, position, shift)
            cropped = img.crop(box)

            # Prepare save path
            name = os.path.basename(img_path)

            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, name)

            cropped.save(save_path)
    print("Images have been cropped successfully!")
