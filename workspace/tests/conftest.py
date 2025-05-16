import pytest
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import numpy as np

@pytest.fixture(scope="session")
def transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@pytest.fixture(scope="session")
def predict(transform):
    def predict_image(model, device, image):
        model.eval()
        with torch.no_grad():
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            return output.argmax(dim=1).item()
    return predict_image

@pytest.fixture(scope="session")
def device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


@pytest.fixture(scope="session")
def model(device):
    model_path = "models/food11.pth"  
    model = torch.load(model_path, map_location=device, weights_only=False)
    _ = model.eval()  
    return model

@pytest.fixture(scope="session")
def test_data(transform):
    from torchvision.datasets import ImageFolder
    food_11_data_dir = os.getenv("FOOD11_DATA_DIR", "Food-11")
    dataset = ImageFolder(os.path.join(food_11_data_dir, 'evaluation'), transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=False)

@pytest.fixture(scope="session")
def predictions(model, device, test_data):
    dataset_size = len(test_data.dataset)
    all_predictions = np.empty(dataset_size, dtype=np.int64)
    all_labels = np.empty(dataset_size, dtype=np.int64)

    current_index = 0
    with torch.no_grad():
        for images, labels in test_data:
            batch_size = labels.size(0)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions[current_index:current_index + batch_size] = predicted.cpu().numpy()
            all_labels[current_index:current_index + batch_size] = labels.cpu().numpy()
            current_index += batch_size

    return all_labels, all_predictions
