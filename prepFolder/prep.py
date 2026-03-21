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
epochs = 5

for epoch in range(5):
    for images, labels in loader:
        output = model(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)


