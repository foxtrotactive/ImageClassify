import os
import torch
import cv2
import numpy as np
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transformers, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# 1 Enviorment setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2 Data preprossessing
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load dataset
dataset = datasets.ImageFolder(root='/path', transform=train_transform)

#split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#3 Model Design with Dropout

model = models.resnet18(pretrained=True) #torch vision library

#Freeze base layers
for param in model.parameters():
    param.requires_grad = False

# Classification head with dropout
model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(dataset,classes))
        )

#set device
model = model.to(device)

#4. Training with Validation

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001) #torch library

# Track metrics
train_losses = []
val_losses = []
train_acc = []
val_acc = []

epochs = 10

for epoch in range(epochs):
    #Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sim().item()

    epoch_train_loss = tunning_loss / len(train_loader.dataset)
    epoch_train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)

    #Validation phase
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

   epoch_val_loss = val_running_loss / len(val_loader.dataset) 
   epoch_val_acc = val_correct . val_total
   val_losses.append(epoch_val_loss)
   val_acc.append(epoch_val_acc)

   print(f'Epoch {epoch+1}/{epochs}')
   print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
   print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}\n')

# 5 Evaluation

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

# Generate confusion matrix

# Make classification report

# 6 Results visualization 
# Loss curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validatioin Loss')
plt.legend()

#Accuracy curve
plt.subplot(1,2,2)
plt.plot(train_acc, label='Train Acc')
plt.plot(val_acc, label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

#todo: results visualization

#todo: get sample predictions and plot images
