import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os


# 1. Dataset Preparation
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# 2. Pretrained ResNet Model
class PretrainedResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PretrainedResNet, self).__init__()
        # Load a pretrained ResNet18 model
        self.resnet = models.resnet18(pretrained=True)
        # Replace the final fully connected layer to match the number of classes
        # The original ResNet18 has `fc` with 512 input features
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def main():
    # Paths to images and labels
    # Labels has been hardcoded. You need to make it through label-studio in a good way.
    image_paths = list(map(lambda x: "classifier_train/" + x, os.listdir("classifier_train")))
    labels = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.03433408, 0.03569807, 0.03586486], std=[0.12090005, 0.12593228, 0.1222396])
    ])

    # Create dataset and dataloader
    dataset = CustomDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize the pretrained ResNet model
    model = PretrainedResNet(num_classes=2)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

    # Save the model
    model_path = "pretrained_resnet_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Evaluation function
    def evaluate_model(model, dataloader):
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
        accuracy = correct / total
        print(f"Accuracy: {accuracy * 100:.2f}%")

    # Evaluate the model
    evaluate_model(model, dataloader)

    # Load the model for later use
    loaded_model = PretrainedResNet(num_classes=2)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.to(device)
    print("Model successfully loaded and ready for use.")


if __name__ == "__main__":
    main()
