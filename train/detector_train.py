import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from torchvision import models

from labels.labels_load import get_coordinates


class GlowDataset(Dataset):
    def __init__(self, image_paths, bounding_boxes, transform=None):
        self.image_paths = image_paths
        self.bounding_boxes = bounding_boxes
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        bbox = self.bounding_boxes[idx]  # (x_min, y_min, x_max, y_max)

        if self.transform:
            image = self.transform(image)

        # Нормализуем координаты (если нужно)
        bbox = torch.tensor(bbox, dtype=torch.float32) / 256.0
        return image, bbox


class GlowDetector(nn.Module):
    def __init__(self):
        super(GlowDetector, self).__init__()
        self.backbone = models.resnet18(pretrained=True)  # Use a pretrained ResNet18
        self.backbone.fc = nn.Identity()  # Remove the classification head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18 outputs 512 features
            nn.ReLU(),
            nn.Linear(256, 4)  # Bounding box regression
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


def train_model(model, dataloader, criterion, optimizer, num_epochs=30):
    loss_data = pd.DataFrame(columns=["epoch", "loss"])
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, bboxes in dataloader:
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, bboxes)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        loss_data = pd.concat([loss_data, pd.DataFrame([[epoch + 1, running_loss/len(dataloader)]],
                                                       columns=loss_data.columns)])
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
    loss_data.to_csv("detector_loss.csv")
    print("Training complete")
    return model


if __name__ == "__main__":
    # Пути к изображениям и координаты bounding box
    image_paths = list(map(lambda x: "detector_train/" + x, os.listdir("detector_train")))
    bounding_boxes = get_coordinates("project-5-at-2025-03-30-14-06-3d3732a9.json") + get_coordinates("project-3-at-2025-03-30-09-01-5b359199.json")

    # Трансформации для изображений
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.03433408, 0.03569807, 0.03586486], std=[0.12090005, 0.12593228, 0.1222396])
    ])

    # Создаем Dataset и DataLoader
    dataset = GlowDataset(image_paths, bounding_boxes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Инициализируем модель, функцию потерь и оптимизатор
    model = GlowDetector()
    criterion = nn.MSELoss()  # Функция потерь для регрессии
    # criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Обучаем модель
    trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=50)

    # Сохраняем модель
    torch.save(trained_model.state_dict(), "glow_detector_resnet.pth")
