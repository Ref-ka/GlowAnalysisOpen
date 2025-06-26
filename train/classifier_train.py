import os
import datetime
import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

from paths import TRAIN_DATA_DIR, MODELS_DIR
from labels.labels_load import get_glow_classes

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class CustomDataset(Dataset):
    """Dataset для классификации свечения."""
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class Classifier(nn.Module):
    """ResNet18-based classifier."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


def collect_dataset(
    dir_names: List[str],
    train_data_dir: str
) -> Tuple[List[str], List[int]]:
    """
    Собирает пути к изображениям и метки классов.
    """
    image_paths = []
    labels = []
    for name in dir_names:
        for dir_type in ("for_detector", "for_predictor"):
            dir_path = os.path.join(train_data_dir, name, "images", dir_type, "prepared")
            label_path = os.path.join(train_data_dir, name, "images", dir_type, "labels.json")
            if not os.path.exists(dir_path) or not os.path.exists(label_path):
                logger.warning(f"Directory or label file not found: {dir_path}, {label_path}")
                continue
            images = [os.path.join(dir_path, image_name) for image_name in os.listdir(dir_path)]
            image_paths.extend(images)
            labels.extend(get_glow_classes(label_path))
    logger.info(f"Total images: {len(image_paths)}")
    return image_paths, labels


def get_default_transform() -> transforms.Compose:
    """
    Возвращает стандартные трансформации для обучения.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.07044805, 0.09651545, 0.07955255],
            std=[0.17199046, 0.18966906, 0.18283128]
        )
    ])


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 50
) -> nn.Module:
    """
    Обучение классификатора свечения.
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.6f}")
    logger.info("Training complete")
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """
    Оценка точности модели на датасете.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def save_model(model: nn.Module, models_dir: str, prefix: str = "resnet") -> str:
    """
    Сохраняет веса модели в указанный каталог.
    """
    save_dir = os.path.join(models_dir, "classifier")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{prefix}_{datetime.datetime.now().date()}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    return save_path


def train_classifier(
        dir_names: List[str],
        train_data_dir: str,
        models_dir: str,
        num_epochs: int = 50,
        batch_size: int = 32,
        num_classes: int = 2,
        learning_rate: float = 0.001
) -> str:
    """
    Обучает классификатор наличия свечения и сохраняет модель.
    Возвращает путь к сохранённой модели.
    """
    image_paths, labels = collect_dataset(dir_names, train_data_dir)
    transform = get_default_transform()
    dataset = CustomDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Classifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        num_epochs=num_epochs
    )

    evaluate_model(trained_model, dataloader, device)
    save_path = save_model(trained_model, models_dir)
    return save_path


if __name__ == "__main__":
    dir_names = [
        "14.04.2025",
        "17.01.2025",
        "11.04.2025",
        "21.01.2025"
    ]
    train_classifier(
        dir_names=dir_names,
        train_data_dir=TRAIN_DATA_DIR,
        models_dir=MODELS_DIR
    )
