import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

torch.manual_seed(1)

dataset = datasets.CIFAR10(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_dataset = datasets.CIFAR10(
    root = './data',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)

loader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle = True
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 1000,
    shuffle = False
)

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64*8*8, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = .001)
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0
    for images,labels in loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total += labels.size(0)
        correct += (output.argmax(1) == labels).sum().item()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_correct += (output.argmax(1) == labels).sum().item()
            test_total += labels.size(0)

    print (f"Epoch: {epoch},Loss: {total_loss/len(loader):.4f},Accuracy: {correct/total:.4f}, Test Acc: {test_correct/test_total:.4f}")

torch.save(model.state_dict(),'model.pth')
