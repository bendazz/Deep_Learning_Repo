import torch

Xm = torch.tensor([
    5.0
])

Xs = torch.tensor([
    4.0
])

Ym = torch.tensor([
    5.0
])

Ys = torch.tensor([
    3.0
])

X_raw = torch.tensor([
    [7.0]
])

X = (X_raw - Xm)/Xs

w = torch.tensor([
    [-.5]
])

b = torch.tensor([
    [0.0]
])

Y = X@w+b

print(Y*Ys + Ym)