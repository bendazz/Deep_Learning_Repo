import torch

X = torch.tensor([
    [2,3]
]).float()

Y = torch.tensor([
    [30]
]).float()

w = torch.tensor([
    [4],
    [5]
]).float()

b = torch.tensor([
    [1]
]).float()

Yhat = X@w + b
r = Yhat - Y
SSE = r.T@r
loss = SSE/1
print(loss.item())