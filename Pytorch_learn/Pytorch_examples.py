import torch
import numpy as np
import torch.nn.functional as F
import Const as C
x1 = torch.tensor(3.0, requires_grad=True)
x2 = torch.tensor(2.0, requires_grad=True)

x3 = torch.tensor(4.0)

y = x1**2 * x2**2 + x3**2

y.backward()



#print(y.grad,x1.grad,x2.grad,x3.grad)

n1 = np.random.random(120).reshape((12,10))

p1 =torch.from_numpy(n1)

n2 = np.random.random(50)

p2 = torch.from_numpy(n2)

n3 = np.random.random(60)

p3 = torch.from_numpy(n3)


x = {"p1":p1,"p2":p2,"p3":p3}

#print(x)

l = [x,x]

#print(l)

import torch.nn as nn

conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, stride= 1,padding=2)

conv2 = nn.Conv2d(in_channels=6, out_channels=12,
                               kernel_size=3, stride=1,padding=1)
pooling1 = nn.MaxPool2d(kernel_size=2)
pooling2 = nn.MaxPool2d(kernel_size=2)

dropout1 = nn.Dropout2d(0.25)
dropout2 = nn.Dropout2d(0.25)
fc_p1 = nn.Linear(in_features=12*10*5, out_features=C.OUT_FEATURES)


input = torch.randn(1,3, 20, 40)

p1 = conv1(input)
p1 = F.relu(p1)
p1 = pooling1(p1)
p1 = dropout1(p1)
p1 = conv2(p1)
p1 = F.relu(p1)
p1 = pooling2(p1)
p1 = dropout2(p1)
print(p1.size())
p1 = p1.view(p1.size(0),-1)
print(p1.size())
p1 = fc_p1(p1)
print(p1.size())

fc_all = torch.nn.Linear(in_features=3 * C.OUT_FEATURES,out_features= 3 )

t = torch.randn(36, 30)
print("t fc_all====>",t.size())
t_all = fc_all(t)
print("fc_all====>",t_all.size())

class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=torch.nn.LSTM(
            input_size=10,
            hidden_size=20,
            num_layers=2,
            batch_first=True
        )
        self.out=torch.nn.Linear(in_features=20,out_features=10)

    def forward(self,x):
        # 以下关于shape的注释只针对单向
        # output: [batch_size, time_step, hidden_size]
        # h_n: [num_layers,batch_size, hidden_size]
        #  虽然LSTM的batch_first为True,但是h_n/c_n的第一维还是num_layers
        # c_n: 同h_n
        output,(h_n,c_n)=self.rnn(x)
        print("ltsm output===>",output.size())
        print("ltsm:h_n===>", h_n.size())
        print("ltsm:c_n===>", c_n.size())
        # output_in_last_timestep=output[:,-1,:] # 也是可以的
        output_in_last_timestep=h_n[-1,:,:]
        print("ltsm:output_in_last_timestep===>", output_in_last_timestep.size())
        # print(output_in_last_timestep.equal(output[:,-1,:])) #ture
        x=self.out(output_in_last_timestep)

        return x


net=RNN()

x = torch.randn(36, 8,10)
print("lstm===>",net(x).size())


loss = nn.CrossEntropyLoss()
# input, NxC=2x3

input = torch.randn(36, 3, requires_grad=True)

# target, N

target = torch.empty(36, dtype=torch.long).random_(3)
output = loss(input, target)



print(input.size())
print(target.size())
print("要计算loss的结果：")
print(output)








