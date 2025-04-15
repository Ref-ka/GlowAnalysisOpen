import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd


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
    transforms.Normalize(mean=[0.03433408, 0.03569807, 0.03586486], std=[0.12090005, 0.12593228, 0.1222396]),
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


def main(image_folder="", model_folder=""):
    paths = []
    labels = []
    for file in os.listdir(image_folder)[:66]:
        paths.append(f"{image_folder}/" + file)
        labels_list = file[:-4].replace("  ", " ").split(" ")
        labels.append(
            [
                float(labels_list[0][:-1].replace(",", ".")),
                float(labels_list[1][:-1].replace(",", "."))
            ]
        )
    print(labels)

    # Data load
    train_dataset = CustomDataset(paths, labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Init model, loss and optimizer
    model = Predictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train
    loss_data = pd.DataFrame(columns=["epoch", "loss"])
    for epoch in range(40):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
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
    torch.save(model.state_dict(), f"{model_folder}/resnet_predictor.pth")


if __name__ == "__main__":
    main(
        "images/prepared",
        "trained_models"
    )
