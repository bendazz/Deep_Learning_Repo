import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from grid import save_image_grid
import numpy as np

dataset = datasets.MNIST(
    root = './data',
    train = True,
    download = True,
    transform = transforms.ToTensor()
)

# image,label = dataset[0]
# image.save('image.png')

# image,label = dataset[0]
# print(image)

loader = DataLoader(
    dataset,
    batch_size = 10
)

for i,(images,labels) in enumerate(loader):
    save_image_grid(images)
    if i == 9:
        break