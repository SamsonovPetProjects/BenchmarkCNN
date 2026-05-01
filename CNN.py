import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import EuroSAT
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Константы

BATCH_SIZE = 32
LEN_OUR_FEATURES = 8
RESNET_RESIZE = 224
RANDOM_ROTATION = 15
COLOR_JITTER = 0.3
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
TRAIN_DATA_PERCENT = 0.8
EPOCH_COUNT = 5
MODEL_SAVE_PATH = "./saved_models"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Валидация + масштабирование

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(RANDOM_ROTATION),
    transforms.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER),
    transforms.Resize((RESNET_RESIZE, RESNET_RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

test_transform = transforms.Compose([
    transforms.Resize((RESNET_RESIZE, RESNET_RESIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

full_dataset = EuroSAT(root='./data', download=True)

class_mapping = {
    0: 0, 6: 0, # Crops
    1: 1,       # Forest
    2: 2, 5: 2, # Grassland
    3: 3,       # Highway
    4: 4,       # Industrial
    7: 5,       # Residential
    8: 6,       # River
    9: 7        # SeaLake
}

new_class_names = [
    "Crops", "Forest", "Natural Grassland", "Highway",
    "Industrial", "Residential", "River", "Sea & Lake"
]

class MappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, mapping, transform=None):
        self.dataset = dataset
        self.mapping = mapping
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image)
        return image, self.mapping[label]

    def __len__(self):
        return len(self.dataset)

train_size = int(TRAIN_DATA_PERCENT * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset_raw, test_dataset_raw = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataset = MappedDataset(train_dataset_raw, class_mapping, transform=train_transform)
test_dataset = MappedDataset(test_dataset_raw, class_mapping, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Фабрика моделей

def create_model(model_name, num_classes):
    if model_name == "ResNet-18":
        model = models.resnet18(weights='IMAGENET1K_V1')
        for param in model.parameters(): param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "ResNet-50":
        model = models.resnet50(weights='IMAGENET1K_V1')
        for param in model.parameters(): param.requires_grad = False
        # Здесь in_features будет равен 2048
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
        for param in model.parameters(): param.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "MobileNet-V3":
        model = models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1')
        for param in model.parameters(): param.requires_grad = False
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    return model

# Функции визуализации

def plot_metrics(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'r-o', label='Loss')
    plt.title(f'{model_name}: Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'g-s', label='Accuracy')
    plt.title(f'{model_name}: Accuracy')
    plt.grid(True)
    plt.show()

def plot_confusion_matrix_heat(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes)
    plt.title(f'Тепловая карта ошибок: {model_name}')
    plt.ylabel('Реальные классы')
    plt.xlabel('Предсказанные классы')
    plt.show()

def plot_roc_auc_curve(all_probs, all_labels, classes, model_name):
    y_bin = label_binarize(all_labels, classes=range(len(classes)))
    plt.figure(figsize=(9, 7))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {auc(fpr, tpr):.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC-AUC Curves: {model_name}')
    plt.legend(loc='lower right')
    plt.show()

# Основный цикл обучения

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_to_test = ["ResNet-18", "ResNet-50", "EfficientNet-B0", "MobileNet-V3"]
results_data = []

for model_name in models_to_test:
    print(f"\n{'='*40}\nЗапуск модели: {model_name}\n{'='*40}")
    
    # Инициализация (всегда новая для чистоты эксперимента)
    model = create_model(model_name, LEN_OUR_FEATURES).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'train_acc': []}
    start_time = time.time()

    # Обучение
    for epoch in range(EPOCH_COUNT):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
        history['train_loss'].append(running_loss/len(train_loader))
        history['train_acc'].append(100. * correct / total)
        print(f"Эпоха {epoch+1}/{EPOCH_COUNT} завершена.")

    train_duration = time.time() - start_time
    
    # Сохранение весов
    safe_name = model_name.replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"{safe_name}.pth"))

    # Оценка
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Метрики
    test_acc = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    # Визуализация для конкретной модели
    plot_metrics(history, model_name)
    plot_confusion_matrix_heat(all_labels, all_preds, new_class_names, model_name)
    plot_roc_auc_curve(np.array(all_probs), np.array(all_labels), new_class_names, model_name)

    results_data.append({
        'Модель': model_name,
        'Accuracy (%)': test_acc,
        'ROC-AUC': macro_auc,
        'Время (сек)': train_duration
    })

# Итоговое сравнение

benchmark_df = pd.DataFrame(results_data)
print("\nИТОГОВАЯ ТАБЛИЦА:")
print(benchmark_df.to_string(index=False))

# Сравнительный график эффективности
plt.figure(figsize=(12, 7))
sns.scatterplot(data=benchmark_df, x='Время (сек)', y='Accuracy (%)', 
                size='ROC-AUC', hue='Модель', sizes=(100, 1000), alpha=0.7)

for i in range(benchmark_df.shape[0]):
    plt.text(benchmark_df['Время (сек)'][i]+1, benchmark_df['Accuracy (%)'][i], 
             benchmark_df['Модель'][i], fontsize=10)

plt.title('Сводный анализ: Точность vs Скорость (размер = ROC-AUC)')
plt.grid(True)
plt.show()
