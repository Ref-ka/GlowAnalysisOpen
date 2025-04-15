from train.predictor_train import Predictor, transform
import torch
from PIL import Image

# Загрузка модели
model = Predictor()
model.load_state_dict(torch.load("trained_models/resnet_predictor.pth"))
model.eval()


# Предсказание
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Добавляем batch dimension
    with torch.no_grad():
        output = model(image)
    return output.numpy()


prediction = predict("prepared/gframe_001200.jpg")
print("Predicted characteristics:", prediction)
print(str(round(prediction[0][0], 1)) + "B")
print(str(round(prediction[0][1], 2)) + "A")
