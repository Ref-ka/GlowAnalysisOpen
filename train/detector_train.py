import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import pandas as pd
from torchvision import models
import datetime
import random

from labels.labels_load import get_coordinates, get_glow_classes
from paths import MODELS_DIR, TRAIN_DATA_DIR
from labels.detector_train_check import visualize_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class AddFakeReflection:
    def __init__(self, probability=0.3):
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            draw = ImageDraw.Draw(img)
            w, h = img.size
            # Random position, small size
            rx, ry = random.randint(0, w-20), random.randint(0, h-20)
            r = random.randint(5, 15)
            draw.ellipse((rx, ry, rx+r, ry+r), fill=(255, 255, 255, 128))
        return img


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
        bbox = torch.tensor(bbox, dtype=torch.float32)  # / 300
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
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


def train_model(model, dataloader, criterion, optimizer, num_epochs=30, dataset_for_vis=None):
    """
    Обучение детектора для обнаружения области свечения на изображении
    :param model:
    :param dataloader:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :param dataset_for_vis:
    :return:
    """
    loss_data = pd.DataFrame(columns=["epoch", "loss"])
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, bboxes in dataloader:
            images = images.to(device)
            bboxes = bboxes.to(device)

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
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.9f}")

        if dataset_for_vis and (epoch + 1) % 20 == 0:
            visualize_predictions(model, dataset_for_vis, epoch + 1, device)

    loss_data.to_csv("detector_loss.csv")
    print("Training complete")
    return model


# Предполагается запускать как самостоятельный скрипт
if __name__ == "__main__":
    # Пути к изображениям и координаты bounding box
    dir_names = ["14.04.2025", "17.01.2025", "11.04.2025", "21.01.2025"]  # Добавляете тут директории с подготовленными данными
    image_paths = []
    bounding_boxes = []
    glow_classes = []

    for name in dir_names:
        for dir_type in ["for_detector", "for_predictor"]:
            dir_path = TRAIN_DATA_DIR + "\\" + name + f"\\images\\{dir_type}\\prepared"
            label_path = TRAIN_DATA_DIR + "\\" + name + f"\\images\\{dir_type}\\labels.json"
            images = list(map(lambda image_name: dir_path + "\\" + image_name, os.listdir(dir_path)))
            boxes = get_coordinates(label_path, 300)  # Указываете размер изображения, который имеют изображения для обучения
            classes = get_glow_classes(label_path)
            image_paths += images
            bounding_boxes += boxes
            glow_classes += classes

    # Filter only images with glow
    filtered_image_paths = []
    filtered_bounding_boxes = []

    for img, box, cls in zip(image_paths, bounding_boxes, glow_classes):
        if cls == 1:  # 1 means "glow"
            filtered_image_paths.append(img)
            filtered_bounding_boxes.append(box)
    print(f"amount of train images: {len(image_paths)}")
    print(f"amount of train filtered images: {len(filtered_image_paths)}")

    # Трансформации для изображений
    transform = transforms.Compose([
        # AddFakeReflection(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.07044805, 0.09651545, 0.07955255], std=[0.17199046, 0.18966906, 0.18283128]) # Эти данные нужно менять при изменении набора обучающих данных
    ])

    # Создаем Dataset и DataLoader
    dataset = GlowDataset(filtered_image_paths, filtered_bounding_boxes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Инициализируем модель, функцию потерь и оптимизатор
    model = GlowDetector().to(device)
    criterion = nn.MSELoss()  # Функция потерь для регрессии
    # criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters())  # , lr=0.0004, weight_decay=1e-4

    # Обучаем модель
    trained_model = train_model(model, dataloader, criterion, optimizer, num_epochs=150, dataset_for_vis=dataset)  # Можно экспериментировать с количеством эпох (num_epoch), но меньше 150 лучше не ставить

    # Сохраняем модель
    torch.save(trained_model.state_dict(), MODELS_DIR + f"\\detector\\resnet_{datetime.datetime.now().date()}.pth")
