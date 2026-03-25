import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image 
from torch.utils.data import DataLoader
from grid import save_image_grid

dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
)

# image,label = dataset[0]
# image.save('image.png')

# image,label = dataset[0]
# save_image(image,'image.png')

loader = DataLoader(
    dataset,
    batch_size = 64,
    shuffle=True
)

test_dataset = datasets.MNIST(
    root = './data',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 1000,
    shuffle=False
)

for i,(image,label) in enumerate(loader):
    save_image_grid(image)
    if i == 9:
        break

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = .001)
epochs = 10

for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        output = model(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == labels).sum().item()
        total += labels.size(0)
    # Test accuracy
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            output = model(images)
            test_correct += (output.argmax(1) == labels).sum().item()
            test_total += labels.size(0)
    model.train()
    print(f"Epoch {epoch+1}  Loss: {total_loss/len(loader):.4f}  Train Acc: {correct/total:.4f}  Test Acc: {test_correct/test_total:.4f}")

torch.save(model.state_dict(), 'mnist_model.pth')
