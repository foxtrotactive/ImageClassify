import os
import torch
import cv2
import numpy as np
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader

# Configuration
num_classes = 10    # Update this with the actual number of product categories
img_size = 640
batch_size = 32
device = 'cuda' if torch.cuda.is_avalable() else 'cpu'

# Custom YOLOv5 model with additional classification layers
class RetailYOLO(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.yolo = base_model
        # Add custom classificatioin head
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0,2),
            nn.Linear(256, num_classes)
                )

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
num_classes = len(train_dataset.classes)  # todo
model = models.resnet18(pretrained=True)

# Freeze parameters and replace final layer
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, num_classes)

# Set device and move model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
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
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {correct/total:.4f}')
