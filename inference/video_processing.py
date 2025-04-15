import cv2
import torch
from PIL import Image
import numpy as np

from image_processing import load_model, preprocess_image, postprocess_detection
from train.classifier_train import PretrainedResNet
from train.detector_train import GlowDetector
from train.predictor_train import Predictor

# Paths to your models
CLASSIFIER_MODEL_PATH = r"..\trained_models\pretrained_resnet_model.pth"
DETECTOR_MODEL_PATH = r"..\trained_models\glow_detector_resnet.pth"
PREDICTOR_MODEL_PATH = r"..\trained_models\resnet_predictor.pth"

# Load models
classifier_model = load_model(CLASSIFIER_MODEL_PATH, PretrainedResNet)
detector_model = load_model(DETECTOR_MODEL_PATH, GlowDetector)
prediction_model = load_model(PREDICTOR_MODEL_PATH, Predictor)


# Video process function
def process_video(video_path, output_video_path, size=256):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (size, size))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}/{total_frames}...")

        # Convert the frame to PIL Image for process
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Preprocess the frame
        image_tensor = preprocess_image(pil_image)

        # Classify the frame
        with torch.no_grad():
            classification_output = classifier_model(image_tensor)
            _, is_glow_present = torch.max(classification_output, 1)
            print(f"no_glow: {int(is_glow_present)}")

        if not int(is_glow_present):
            original_image = np.array(pil_image)
            processed_frame = original_image
            # Format the predicted parameters for display
            predicted_text = [
                "No glow"
            ]

            # Overlay the predicted parameters on the processed frame
            for i, text in enumerate(predicted_text):
                cv2.putText(
                    processed_frame,  # Frame to draw on
                    text,  # Text to display
                    (10, 30 + i * 30),  # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font
                    0.5,  # Font scale
                    (255, 255, 255),  # Font color (white)
                    1,  # Thickness
                    cv2.LINE_AA  # Line type
                )
        else:
            # Detect glow areas
            with torch.no_grad():
                detection_output = detector_model(image_tensor)  # Assuming detector returns bounding boxes
                print(f"Frame {frame_count} detection output: {detection_output}")

            # Postprocess detection
            original_image = np.array(pil_image)
            processed_frame = postprocess_detection(original_image, detection_output, size)

            # Predict parameters using the prediction model
            with torch.no_grad():
                prediction_output = prediction_model(image_tensor).numpy()

            # Format the predicted parameters for display
            predicted_text = [
                f"{round(float(prediction_output[0][0]), 1)} B",
                f"{round(float(prediction_output[0][1]), 2)} A",
            ]

            # Overlay the predicted parameters on the processed frame
            for i, text in enumerate(predicted_text):
                cv2.putText(
                    processed_frame,  # Frame to draw on
                    text,  # Text to display
                    (10, 30 + i * 30),  # Position (x, y)
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font
                    0.5,  # Font scale
                    (255, 255, 255),  # Font color (white)
                    1,  # Thickness
                    cv2.LINE_AA  # Line type
                )

        # Convert processed frame back to BGR for saving
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

        # Write the processed frame to the output video
        out.write(processed_frame_bgr)

    # Release resources
    cap.release()
    out.release()
    print("Video process complete. Output saved to:", output_video_path)


if __name__ == "__main__":
    input_video_path = r"..\videos\1_prepared.avi"
    output_video_path = r"..\videos\output_video_2_feat.avi"

    process_video(input_video_path, output_video_path)
