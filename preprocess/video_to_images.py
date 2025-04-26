import cv2
import os
from pathlib import Path


def extract_frames(video_path, output_dir, frame_rate=1):
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

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame if it matches the frame rate condition
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{Path(video_path).name}_{frame_count:06d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete. {saved_count} frames saved to {output_dir}.")


def cut_videos(video_dir: str, output_dir: str, frame_rate=1):
    for video_name in os.listdir(video_dir):
        extract_frames(video_dir + "\\" + video_name, output_dir, frame_rate)
    print("Videos have been cut successfully!!!")


if __name__ == "__main__":
    pass
