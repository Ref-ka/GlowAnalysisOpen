import cv2
import os


def extract_frames(video_path, output_folder, frame_rate=1):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
            break  # Exit the loop if no more frames are available

        # Save the frame if it matches the frame rate condition
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_1_2_{frame_count:06d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete. {saved_count} frames saved to {output_folder}.")


# Example usage
if __name__ == "__main__":
    video_file = "videos/1_2.avi"  # Replace with your .avi video file path
    output_dir = "unprepared_images"  # Replace with your desired output folder
    frame_skip = 20  # Save every frame (set to 2 to save every second frame, etc.)

    extract_frames(video_file, output_dir, frame_skip)
