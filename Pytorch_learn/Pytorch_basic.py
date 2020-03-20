import torch
import numpy as np


x1 = torch.tensor(3.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)

x3 = torch.tensor(4.0)

y = x1**2 * x2**2 + x3**2

y.backward()



print(y.grad,x1.grad,x2.grad,x3.grad)

n1 = np.random.random(120).reshape((12,10))

p1 =torch.from_numpy(n1)

n2 = np.random.random(50)

p2 = torch.from_numpy(n2)

n3 = np.random.random(60)

p3 = torch.from_numpy(n3)


x = {"p1":p1,"p2":p2,"p3":p3}

print(x)

l = [x,x]

print(l)

