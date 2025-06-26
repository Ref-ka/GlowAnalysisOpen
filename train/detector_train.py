import os
import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from torchvision import models
import datetime

from labels.labels_load import get_coordinates, get_glow_classes
from paths import MODELS_DIR, TRAIN_DATA_DIR
from labels.detector_train_check import visualize_predictions

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class GlowDataset(Dataset):
    """Dataset для обучения детектора свечения."""

    def __init__(
            self,
            image_paths: List[str],
            bounding_boxes: List[Tuple[float, float, float, float]],
            transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        bbox = self.bounding_boxes[idx]

        if self.transform:
            image = self.transform(image)

        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        return image, bbox_tensor


class Detector(nn.Module):
    """ResNet18-based bounding box regressor."""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.fc(x)
        return x


def train_model(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int = 30,
        dataset_for_vis: Optional[Dataset] = None,
        loss_csv_path: str = "detector_loss.csv"
) -> nn.Module:
    """
    Обучение детектора для обнаружения области свечения на изображении.
    """
    loss_data = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, bboxes in dataloader:
            images = images.to(DEVICE)
            bboxes = bboxes.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, bboxes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_data.append({"epoch": epoch + 1, "loss": epoch_loss})
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.9f}")

        if dataset_for_vis and (epoch + 1) % 20 == 0:
            visualize_predictions(model, dataset_for_vis, epoch + 1, DEVICE)

    pd.DataFrame(loss_data).to_csv(loss_csv_path, index=False)
    logger.info("Training complete")
    return model


def collect_dataset(
        dir_names: List[str],
        train_data_dir: str,
        image_size: int = 300
) -> Tuple[List[str], List[Tuple[float, float, float, float]]]:
    """
    Собирает пути к изображениям и bounding boxes только для классов 'glow'.
    """
    image_paths = []
    bounding_boxes = []
    glow_classes = []

    for name in dir_names:
        for dir_type in ("for_detector", "for_predictor"):
            dir_path = os.path.join(train_data_dir, name, "images", dir_type, "prepared")
            label_path = os.path.join(train_data_dir, name, "images", dir_type, "labels.json")
            if not os.path.exists(dir_path) or not os.path.exists(label_path):
                logger.warning(f"Directory or label file not found: {dir_path}, {label_path}")
                continue
            images = [os.path.join(dir_path, image_name) for image_name in os.listdir(dir_path)]
            boxes = get_coordinates(label_path, image_size)
            classes = get_glow_classes(label_path)
            image_paths.extend(images)
            bounding_boxes.extend(boxes)
            glow_classes.extend(classes)

    filtered_image_paths = []
    filtered_bounding_boxes = []
    for img, box, cls in zip(image_paths, bounding_boxes, glow_classes):
        if cls == 1:  # 1 means "glow"
            filtered_image_paths.append(img)
            filtered_bounding_boxes.append(box)

    logger.info(f"Total images: {len(image_paths)}")
    logger.info(f"Filtered (glow) images: {len(filtered_image_paths)}")
    return filtered_image_paths, filtered_bounding_boxes


def get_default_transform() -> transforms.Compose:
    """
    Возвращает стандартные трансформации для обучения.
    """
    return transforms.Compose([
        # AddFakeReflection(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.07044805, 0.09651545, 0.07955255],
            std=[0.17199046, 0.18966906, 0.18283128]
        )
    ])


def save_model(model: nn.Module, models_dir: str, prefix: str = "resnet") -> str:
    """
    Сохраняет веса модели в указанный каталог.
    """
    os.makedirs(os.path.join(models_dir, "detector"), exist_ok=True)
    filename = f"{prefix}_{datetime.datetime.now().date()}.pth"
    save_path = os.path.join(models_dir, "detector", filename)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    return save_path


def train_detector(
        dir_names: List[str],
        train_data_dir: str,
        models_dir: str,
        num_epochs: int = 150,
        batch_size: int = 32,
        image_size: int = 300,
        loss_csv_path: str = "detector_loss.csv",
        model_prefix: str = "resnet18",
        learning_rate: float = 0.001
) -> str:
    """
    Обучает детектор свечения и сохраняет модель.
    Возвращает путь к сохранённой модели.
    """
    image_paths, bounding_boxes = collect_dataset(dir_names, train_data_dir, image_size=image_size)
    transform = get_default_transform()
    dataset = GlowDataset(image_paths, bounding_boxes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Detector().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        dataset_for_vis=dataset,
        loss_csv_path=loss_csv_path
    )

    save_path = save_model(trained_model, models_dir, prefix=model_prefix)
    return save_path


if __name__ == "__main__":
    dir_names = [
        "14.04.2025",
        "17.01.2025",
        "11.04.2025",
        "21.01.2025"
    ]
    train_detector(
        dir_names=dir_names,
        train_data_dir=TRAIN_DATA_DIR,
        models_dir=MODELS_DIR,
        num_epochs=150,
        batch_size=32,
        image_size=300
    )
