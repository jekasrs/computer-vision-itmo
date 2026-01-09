import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report

from Lab_3.model.model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Преобразования
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    "data/test",
    transform=transform
)

class_names = test_dataset.classes  # ['hamburger', 'pizza', 'sushi']

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False
)

# Загрузка обученной модели
model = SimpleCNN(num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("model/cnn_food101.pth", map_location=device))
model.eval()

y_true = []
y_pred = []

# Предсказания
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# 1. Accuracy по каждому классу
print("Точность по каждому классу:\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

# 2. Общая accuracy
overall_accuracy = (y_true == y_pred).mean()
print(f"Общая точность (Accuracy): {overall_accuracy:.4f}")

# 3. Матрица ошибок
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(
    ticks=np.arange(len(class_names)),
    labels=class_names,
    rotation=45
)
plt.yticks(
    ticks=np.arange(len(class_names)),
    labels=class_names
)

plt.xlabel("Predicted label")
plt.ylabel("True label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

plt.tight_layout()
plt.show()