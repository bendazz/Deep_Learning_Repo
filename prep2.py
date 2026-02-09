import torch

X = torch.tensor([
    [2.0],
    [5.0],
    [10.0]
])

Y = torch.tensor([
    [9.0],
    [1.0],
    [3.0]
])

meanX = X.mean()
stdX = X.std()
meanY = Y.mean()
stdY = Y.std()

X = (X - meanX)/stdX
Y = (Y - meanY)/stdY

w = torch.tensor([
    [0.0]
],requires_grad = True)

b = torch.tensor([
    [0.0]
],requires_grad = True)

epochs = 10000
lr = .01

for epoch in range(epochs):
    Yhat = X@w + b
    r = Y - Yhat
    loss = r.T@r/3
    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad  
        b -= lr*b.grad  

    w.grad.zero_()
    b.grad.zero_()

    print(loss.item(),w,b)

    prediction = w*(7-meanX)/stdX + b
    print(prediction*stdY+meanY)

    #3.4898