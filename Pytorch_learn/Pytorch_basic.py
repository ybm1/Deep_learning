import torch

x1 = torch.tensor(3.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)

x3 = torch.tensor(4.0)

y = x1**2 * x2**2 + x3**2

y.backward()



print(y.grad,x1.grad,x2.grad,x3.grad)




