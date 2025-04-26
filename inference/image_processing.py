import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

from train.classifier_train import PretrainedResNet
from train.detector_train import GlowDetector


# Load models
def load_model(model_path, model_class):
    # Initialize the model architecture
    model = model_class()

    # Load the state dictionary
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)

    # Set the model to evaluation mode
    model.eval()

    return model


# Preprocessing function
def preprocess_image(image_input):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.04102539, 0.13573588, 0.08836696], std=[0.1449014, 0.22010248, 0.19639405])
    ])
    if isinstance(image_input, str):  # If it's a file path
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):  # If it's a PIL.Image object
        image = image_input.convert('RGB')
    else:
        raise ValueError("Invalid input type. Expected a file path or a PIL.Image object.")
    return transform(image).unsqueeze(0)  # Add batch dimension


# Postprocessing function for detector output
def postprocess_detection(image, detection_output, size=256):
    # Assuming detection_output contains bounding boxes in the format [x1, y1, x2, y2]
    black_image = np.zeros((size, size, 3), dtype=np.uint8)
    detection_output = detection_output * size
    for box in detection_output:
        x1, y1, x2, y2 = map(int, box)
        glow_area = image[y1:y2, x1:x2]

        # Get the dimensions of the glow_area
        glow_h, glow_w, _ = glow_area.shape

        # Calculate the top-left corner of where to place the glow_area in the black_image
        start_y = (size - glow_h) // 2
        start_x = (size - glow_w) // 2

        # Ensure the glow_area fits within the black_image boundaries
        if start_y < 0 or start_x < 0 or start_y + glow_h > size or start_x + glow_w > size:
            raise ValueError("Glow area is too large to fit in the black image without resizing.")

        # Place the glow_area into the black_image
        black_image[start_y:start_y + glow_h, start_x:start_x + glow_w] = glow_area

    return black_image


# Main preprocess function
def process_images(image_dir, classifier_model, detector_model, output_dir, size=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = preprocess_image(image_path)

        # Classify image
        with torch.no_grad():
            classification_output = classifier_model(image)
            is_glow_present = torch.max(classification_output, 1)

        if not is_glow_present:
            # Replace with black image
            black_image = np.zeros((size, size, 3), dtype=np.uint8)
            black_image = Image.fromarray(black_image)
            black_image.save(os.path.join(output_dir, image_name))
        else:
            # Detect glow areas
            with torch.no_grad():
                detection_output = detector_model(image)  # Assuming detector returns bounding boxes
            original_image = np.array(Image.open(image_path).convert('RGB'))
            processed_image = postprocess_detection(original_image, detection_output, size)
            processed_image = Image.fromarray(processed_image)
            processed_image.save(os.path.join(output_dir, image_name))


if __name__ == "__main__":
    classifier_model_path = "simple_cnn_model.pth"
    detector_model_path = "glow_detector_resnet.pth"
    input_image_dir = "detector_train"
    output_image_dir = "images/prepared"

    classifier_model = load_model(classifier_model_path, PretrainedResNet)
    detector_model = load_model(detector_model_path, GlowDetector)

    process_images(input_image_dir, classifier_model, detector_model, output_image_dir)