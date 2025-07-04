import os
import logging
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import datetime

from paths import MODELS_DIR, TRAIN_DATA_DIR

# Настройка логгирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")


class PredictorDataset(Dataset):
    """Dataset для обучения предиктора (регрессия по двум признакам)."""
    def __init__(
        self,
        image_paths: List[str],
        labels: List[Tuple[float, float]],
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor


class Predictor(nn.Module):
    """ResNet18-based регрессор для двух признаков."""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


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


def collect_dataset(
    dir_names: List[str],
    train_data_dir: str
) -> Tuple[List[str], List[Tuple[float, float]]]:
    """
    Собирает пути к изображениям и метки для обучения предиктора.
    """
    image_paths = []
    labels = []

    for name in dir_names:
        dir_path = os.path.join(train_data_dir, name, "images", "for_predictor", "extracted")
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found: {dir_path}")
            continue
        for image_name in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_name)
            image_paths.append(image_path)
            # Извлекаем метки из имени файла
            base = os.path.splitext(image_name)[0]
            label_strs = base.replace("_", " ").split(" ")
            if len(label_strs) < 2:
                logger.warning(f"Cannot parse label from filename: {image_name}")
                continue
            try:
                label = (float(label_strs[0]), float(label_strs[1]))
            except ValueError:
                logger.warning(f"Cannot convert label to float: {image_name}")
                continue
            labels.append(label)
    logger.info(f"Total images: {len(image_paths)}")
    return image_paths, labels


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = 40,
    loss_csv_path: str = "predictor_loss.csv"
) -> nn.Module:
    """
    Обучение предиктора.
    """
    loss_data = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_data.append({"epoch": epoch + 1, "loss": epoch_loss})
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.9f}")

    pd.DataFrame(loss_data).to_csv(loss_csv_path, index=False)
    logger.info("Training complete")
    return model


def evaluate_mse(model: nn.Module, data_loader: DataLoader, device):
    model.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = mse_loss(preds, targets)
            total_loss += loss.item()
            total_samples += images.size(0)
    return total_loss / total_samples


def save_model(model: nn.Module, models_dir: str, prefix: str = "resnet") -> str:
    """
    Сохраняет веса модели в указанный каталог.
    """
    save_dir = os.path.join(models_dir, "predictor")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{prefix}_{datetime.datetime.now().date()}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")
    return save_path


def train_predictor(
        dir_names: List[str],
        train_data_dir: str,
        models_dir: str,
        num_epochs: int = 40,
        batch_size: int = 32,
        model_prefix: str = "resnet18",
        learning_rate: float = 0.001,
        loss_csv_path: str = "predictor_loss.csv"
) -> str:
    """
    Обучает предиктор характеристик свечения и сохраняет модель.
    Возвращает путь к сохранённой модели.
    """
    image_paths, labels = collect_dataset(dir_names, train_data_dir)
    transform = get_default_transform()
    dataset = PredictorDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Predictor().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trained_model = train_model(
        model,
        dataloader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        loss_csv_path=loss_csv_path
    )

    mse = evaluate_mse(trained_model, dataloader, DEVICE)
    logger.info(f"MSE: {mse:.8f}")

    save_path = save_model(trained_model, models_dir, prefix=model_prefix)
    return save_path


if __name__ == "__main__":
    dir_names = [
        "11.04.2025",
        "12.04.2025",
        "13.04.2025",
        "14.04.2025",
        "17.01.2025",
        "21.01.2025"
    ]
    train_predictor(
        dir_names=dir_names,
        train_data_dir=TRAIN_DATA_DIR,
        models_dir=MODELS_DIR,
        num_epochs=100
    )
