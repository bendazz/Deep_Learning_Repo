import torch

X = torch.tensor([[3.0]])
Y = torch.tensor([[2.0]])
w = torch.tensor([[5.0]],requires_grad = True)
b = torch.tensor([[1.0]],requires_grad = True)
lr = .2

Yhat = X@w+b
r = Yhat - Y  
SSE = r.T@r 
loss = r.T@r/1   


loss.backward()

with torch.no_grad():
    w -= lr*w.grad
    b -= lr*b.grad

w.grad.zero_()
b.grad.zero_()

print(w,b)

