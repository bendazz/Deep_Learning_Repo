import torch  

x = torch.tensor(3.0,requires_grad = True)
y = torch.tensor(-1.0,requires_grad = True)
z = torch.tensor(0.0,requires_grad = True)

f = -3*z**3*y**3 +3*x +2*z
f.backward()
print(x.grad,y.grad,z.grad)




