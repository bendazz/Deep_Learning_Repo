import torch

x = torch.tensor(3.0,requires_grad = True)
f = x**2
f.backward()
print(x.grad)


x.grad.zero_()
f = x**3
f.backward()
print(x.grad)