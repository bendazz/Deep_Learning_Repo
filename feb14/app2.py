import torch

X_raw = torch.tensor([
    [1.0],
    [5.0],
    [9.0]
])

Y_raw = torch.tensor([
    [5.0],
    [8.0],
    [2.0]
])

Xm = X_raw.mean()
Xs = X_raw.std()

Ym = Y_raw.mean()
Ys = Y_raw.std()

X = (X_raw - Xm)/Xs

Y = (Y_raw - Ym)/Ys

w = torch.tensor([
    [0.0]
],requires_grad = True)

b = torch.tensor([
    [0.0]
],requires_grad = True)

epochs = 1000
lr = .01

for epoch in range(epochs):
    Yhat = X@w + b
    r = Yhat - Y
    loss = r.T@r/3
    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad

    w.grad.zero_()
    b.grad.zero_()

    print(loss.item(),w,b)