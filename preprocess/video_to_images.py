import cv2
import os
from pathlib import Path
import numpy as np


def extract_frames(video_path, output_dir, n_images=10):
    """
    Функция нарезания видео для обучения на изображения
    :param video_path:
    :param output_dir:
    :param n_images:
    :return:
    """
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in video: {total_frames}")

    if n_images > total_frames:
        n_images = total_frames  # не больше, чем есть кадров

    frame_indices = np.linspace(0, total_frames - 1, n_images, dtype=int)

    saved_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Couldn't read frame {idx}")
            continue
        frame_filename = os.path.join(
            output_dir, f"frame_{Path(video_path).stem}_{int(idx):06d}.png"
        )
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    cap.release()
    print(f"Extraction complete. {saved_count} frames saved to {output_dir}.")

    cap.release()
    print(f"Extraction complete. {saved_count} frames saved to {output_dir}.")


def cut_videos(video_dir: str, output_dir: str, n_images: int):
    """
    Вспомогательная функция для обработки нескольких изображений
    :param video_dir:
    :param output_dir:
    :param n_images:
    :return:
    """
    for video_name in os.listdir(video_dir):
        extract_frames(video_dir + "\\" + video_name, output_dir, n_images)
    print("Videos have been cut successfully!!!")


# Используется автоматически в prepare images
if __name__ == "__main__":
    pass
