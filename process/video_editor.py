from moviepy import VideoFileClip


def crop_video(input_path, output_path, crop_width, crop_height):
    # Load the video
    video = VideoFileClip(input_path)

    # Calculate the center of the video
    video_width, video_height = video.size
    center_x, center_y = video_width // 2 + 30, video_height // 2 - 130

    # Calculate cropping box
    x1 = center_x - crop_width // 2
    y1 = center_y - crop_height // 2
    x2 = center_x + crop_width // 2
    y2 = center_y + crop_height // 2

    # Crop the video
    cropped_video = video.cropped(x1=x1, y1=y1, x2=x2, y2=y2)

    # Write the cropped video to the output file
    cropped_video.write_videofile(output_path, codec="libx264", audio_codec="aac")


# Input and output paths
input_video_path = "videos/1.avi"  # Replace with your input video file path
output_video_path = "videos/1_prepared.avi"  # Replace with your desired output file path

# Crop dimensions
crop_width = 256
crop_height = 256

# Crop the video
crop_video(input_video_path, output_video_path, crop_width, crop_height)