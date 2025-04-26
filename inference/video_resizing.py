import cv2
from PIL import Image
import numpy as np

from preprocess.image_resizing import get_crop_box
from paths import TESTING_DATA_DIR


def crop_video(
        input_path: str,
        output_path: str,
        crop_size: int,
        position: str,
        shift=None
):
    """
    Crop each frame of a video using the same logic as crop_images.

    Args:
        input_path (str): Path to input video.
        output_path (str): Path to save output video.
        crop_size (int): Size of the square crop.
        position (str): Crop position ("l", "lu", "c", "b", etc.).
        shift (list[int] or None): shift for images by X and Y axes
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame of video.")

    # Get frame size
    img_h, img_w = frame.shape[:2]
    if crop_size > img_w or crop_size > img_h:
        raise ValueError(f"Crop size {crop_size} is larger than video frame size {img_w}x{img_h}")

    # Prepare video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_size, crop_size))

    frame_idx = 0
    while ret:
        # Convert frame (numpy array) to PIL Image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        box = get_crop_box(img_w, img_h, crop_size, position, shift)
        cropped = img.crop(box)
        # Convert back to OpenCV format
        cropped_cv = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
        out.write(cropped_cv)
        ret, frame = cap.read()
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video cropped and saved to {output_path}")


if __name__ == "__main__":
    crop_video(TESTING_DATA_DIR + "\\videos\\17.01.2025\\vid_3.avi",
               TESTING_DATA_DIR + "\\videos\\17.01.2025\\vid_3_processed.avi",
               300,
               "c",
               [60, -120])
