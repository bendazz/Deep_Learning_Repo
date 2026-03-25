import torch
from torchvision import datasets,transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

torch.manual_seed(1)

dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

test_dataset = datasets.MNIST(
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

DataLoader(
    test_dataset,
    batch_size = 1000,
    shuffle = False
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)
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
    print (f"Epoch: {epoch},Loss: {total_loss/len(loader)},Accuracy: {correct/total}")
