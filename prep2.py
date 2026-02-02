import pandas as pd
import torch

df = pd.read_csv("data.csv")
X = torch.tensor(df.drop("Y",axis = 1).to_numpy()).float()
Y = torch.tensor(df["Y"].to_numpy()).float().reshape(-1,1)

Xmean = X.mean()
Xstd = X.std()
Ymean = Y.mean()
Ystd = Y.std()

X = (X - Xmean)/Xstd
Y = (Y - Ymean)/Ystd

w = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [0.0]
],requires_grad = True)
b = torch.tensor([[0.0]],requires_grad = True)

epochs = 1000000
lr = .1

for epoch in range(epochs):
    Yhat = X@w + b
    r = Yhat - Y  
    loss = r.T@r/7
    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad 
        b -= lr*b.grad 

    w.grad.zero_()
    b.grad.zero_()
    #print(loss,w,b) 

X = torch.tensor([
    [7.5,15,70,1]
])
X = (X-Xmean)/Xstd
Y = X@w+b
print(Y*Ystd+Ymean)









