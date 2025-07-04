import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

from paths import MODELS_DIR, TRAIN_DATA_DIR
from train.classifier_train import Classifier
from train.detector_train import Detector


# Загрузка моделей
def load_model(model_path, model_class):
    # Initialize the model architecture
    model = model_class()

    # Load the state dictionary
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)

    # Set the model to evaluation mode
    model.eval()

    return model


# Предобработка во время обучения
def preprocess_image(image_input):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.07044805, 0.09651545, 0.07955255], std=[0.17199046, 0.18966906, 0.18283128])  # Эти значения нужно менять в соответствии с набором обучающих данных
    ])
    if isinstance(image_input, str):  # If it's a file path
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):  # If it's a PIL.Image object
        image = image_input.convert('RGB')
    else:
        raise ValueError("Invalid input type. Expected a file path or a PIL.Image object.")
    return transform(image).unsqueeze(0)  # Add batch dimension


# Постобработка предсказаний детектора
def postprocess_detection(image, detection_output, image_size: int = 300, normalization: bool = False):
    # Assuming detection_output contains bounding boxes in the format [x1, y1, x2, y2]
    black_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    detection_output = detection_output * image_size if normalization else detection_output
    for box in detection_output:
        x1, y1, x2, y2 = [float(v) for v in box]

        x1 = max(0, min(int(round(x1, 3)), image_size - 1))
        y1 = max(0, min(int(round(y1, 3)), image_size - 1)) - 5
        x2 = max(0, min(int(round(x2, 3)), image_size))
        y2 = max(0, min(int(round(y2, 3)), image_size))
        glow_area = image[y1:y2, x1:x2]

        # Get the dimensions of the glow_area
        glow_h, glow_w, _ = glow_area.shape

        # Calculate the top-left corner of where to place the glow_area in the black_image
        start_y = (image_size - glow_h) // 2
        start_x = (image_size - glow_w) // 2

        # Ensure the glow_area fits within the black_image boundaries
        if start_y < 0 or start_x < 0 or start_y + glow_h > image_size or start_x + glow_w > image_size:
            raise ValueError("Glow area is too large to fit in the black image without resizing.")

        # Place the glow_area into the black_image
        black_image[y1:y1 + glow_h, x1:x1 + glow_w] = glow_area

    return black_image


def process_images(image_dir, classifier_model, detector_model, output_dir, size=300):
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


def prepare_predictor_images(classifier_name: str, detector_name: str, dir_list: list[str]):
    classifier_path = MODELS_DIR + f"\\classifier\\{classifier_name}"
    detector_path = MODELS_DIR + f"\\detector\\{detector_name}"

    dir_list = list(map(lambda dir_name: TRAIN_DATA_DIR + "\\" + dir_name, dir_list))
    for i, directory in enumerate(dir_list):
        process_images(
            directory + "\\images\\for_predictor\\prepared",
            load_model(classifier_path, Classifier),
            load_model(detector_path, Detector),
            directory + "\\images\\for_predictor\\extracted"
        )


if __name__ == "__main__":
    # Используется для подготовки изображений перед обучением предиктора
    prepare_predictor_images(
        "resnet_2025-04-30.pth",
        "resnet18_2025-07-03.pth",
        [
            "11.04.2025",
            "12.04.2025",
            "13.04.2025",
            "14.04.2025",
            "17.01.2025",
            "21.01.2025"
        ]
    )
