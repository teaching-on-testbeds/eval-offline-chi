import os
import random
from PIL import Image

# --- External Datasets ---

CAKE_DIR = "cake_looks_like"
INDIAN_DIR = "indian_dessert"


def evaluate_folder(model, device, folder_path, predict):
    correct = 0
    total = 0
    for fname in os.listdir(folder_path):
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        pred = predict(model, device, img)
        if pred == 2:  # 'Dessert' index
            correct += 1
        total += 1
    return correct / total * 100 if total > 0 else 0

# Require 60% accuracy
def test_cake_looks_like_accuracy(model, device, predict):
    acc = evaluate_folder(model, device, CAKE_DIR, predict)
    assert acc >= 60, f"{CAKE_DIR} accuracy too low: {acc:.2f}%"

# Require 60% accuracy
def test_indian_dessert_accuracy(model, device, predict):
    acc = evaluate_folder(model, device, INDIAN_DIR, predict)
    assert acc >= 60, f"{INDIAN_DIR} accuracy too low: {acc:.2f}%"

