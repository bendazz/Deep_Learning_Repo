import torch

X = torch.tensor([
    7.0
])

w = torch.tensor([
    [-.375]
])

b = torch.tensor([
    [6.875]
])

print(X@w+b)