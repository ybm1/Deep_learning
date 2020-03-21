from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

import Const as C
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                               kernel_size=5, stride= 1,padding=2)
        # 上一个con的out_channels要和下一个conv2的in_channels相等
        # padding = (kernel_size-1)//2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12,
                               kernel_size=3, stride=1,padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)

        self.pooling1 = nn.MaxPool2d(kernel_size=2)
        self.pooling2 = nn.MaxPool2d(kernel_size=2)

        self.fc_p1 = nn.Linear(in_features=12*5*10, out_features=C.OUT_FEATURES)

        self.lstm = torch.nn.LSTM(
            input_size=C.INPUT_SIZE,
            hidden_size=C.HIDDEN_SIZE,
            num_layers=C.NUM_LAYERS,
            batch_first=True
        )

        self.fc_p2 = torch.nn.Linear(in_features=C.HIDDEN_SIZE, out_features=C.OUT_FEATURES)

        self.fc_p3 = torch.nn.Linear(in_features= 10, out_features=C.OUT_FEATURES)

        self.fc_all = torch.nn.Linear(in_features=3*C.OUT_FEATURES, out_features=1)

    def forward(self, p1,p2,p3):
        p1 = self.conv1(p1)
        p1 = F.relu(p1)
        p1 = self.pooling1(p1)
        p1 = self.dropout1(p1)
        p1 = self.conv2(p1)
        p1 = F.relu(p1)
        p1 = self.pooling2(p1)
        p1 = self.dropout2(p1)
        p1 = p1.view(p1.size(0),-1)
        p1 = self.fc_p1(p1)


        p2,(h_n,c_n) = self.lstm(p2)
        output_in_last_timestep = h_n[-1, :, :]
        p2 = self.fc_p2(output_in_last_timestep)


        p3 = self.fc_p3(p3)
        p3 = F.sigmoid(p3)

        ##print("model===>",p1.size(),p2.size(),p3.size())

        p_all = torch.cat((p1,p2,p3),1)
        output = self.fc_all(p_all)
        output = F.sigmoid(output)

        return output


if __name__ == '__main__':
    model = Net()
    print(model)




