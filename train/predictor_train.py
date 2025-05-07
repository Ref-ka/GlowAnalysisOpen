import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd
import datetime

from paths import MODELS_DIR, TRAIN_DATA_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Defining the model
class Predictor(nn.Module):
    def __init__(self):
        super(Predictor, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)

    def forward(self, x):
        return self.base_model(x)


# Processing train images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.07044805, 0.09651545, 0.07955255], std=[0.17199046, 0.18966906, 0.18283128]),  # Изменяется в соответствии с набором обучающих данных
])


# 3. Создание кастомного Dataset
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float)


def main():
    # TODO: Сделать отдельную функцию для обучения,
    #  сделать количество признаков опциональным

    # Процесс обучения предиктора
    dir_names = ["14.04.2025", "17.01.2025", "11.04.2025", "21.01.2025"]
    image_paths = []
    labels = []
    for name in dir_names:
        dir_path = TRAIN_DATA_DIR + "\\" + name + f"\\images\\for_predictor\\prepared"
        image_paths += list(map(lambda image_name: dir_path + "\\" + image_name, os.listdir(dir_path)))

    for path in image_paths:
        path = path.split("\\")[-1]
        labels_list = path[:-4].replace("_", " ").split(" ")
        labels.append(
            [
                float(labels_list[0]),
                float(labels_list[1])
            ]
        )

    # Data load
    train_dataset = CustomDataset(image_paths, labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Init model, loss and optimizer
    model = Predictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    loss_data = pd.DataFrame(columns=["epoch", "loss"])
    for epoch in range(40):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss_data = pd.concat([loss_data, pd.DataFrame([[epoch + 1, running_loss / len(train_loader)]],
                                                       columns=loss_data.columns)])
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    loss_data.to_csv("predictor_loss.csv")

    # Save model
    torch.save(model.state_dict(), MODELS_DIR + f"\\predictor\\resnet_{datetime.datetime.now().date()}.pth")


if __name__ == "__main__":
    main()
