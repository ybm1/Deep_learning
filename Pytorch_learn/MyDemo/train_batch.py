from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import Net
from multi_input_data import get_data,Mydataset
from torch.utils.data import DataLoader
import Const as C

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(C.PATH_to_log_dir)

## 使用
# tensorboard --logdir=/Users/biqixuan/PycharmProjects/Deep_learning/Pytorch_learn/MyDemo/logs

# 进行tensorboard的查看


def mytrain(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(train_loader):
        p1 = sample_batched["features"]["p1"]
        p2 = sample_batched["features"]["p2"]
        p3 = sample_batched["features"]["p3"]

        target = sample_batched["label"]

        p1,p2,p3,target = p1.to(device),p2.to(device),p3.to(device),target.to(device)

        optimizer.zero_grad()
        output = model(p1,p2,p3)

        loss = F.mse_loss(output, target)

        loss.backward()

        optimizer.step()
        bias = F.l1_loss(output, target)

        writer.add_scalar('training MSE',
                          loss.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('training l1 loss',
                          bias.item(),
                          epoch * len(train_loader) + batch_idx)


        if batch_idx % C.LOG_INTERVAL == 0:
            print('Train Epoch: {} Batch_idx :{} \tLoss: {:.6f}'.format(
                epoch, batch_idx , loss.item()))


def mytest(model, device, test_loader,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            p1 = data["features"]["p1"]
            p2 = data["features"]["p2"]
            p3 = data["features"]["p3"]

            target = data["label"]

            p1, p2, p3, target = p1.to(device), p2.to(device), p3.to(device), target.to(device)

            output = model(p1,p2,p3)
            test_loss += F.mse_loss(input = output, target = target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    writer.add_scalar('Epoch test MSE',
                      test_loss,epoch)

    print('\nTest set:Epoch: {}  Average MSE loss: {:.4f}'.format(epoch,test_loss))




if __name__ == '__main__':
    train_input = Mydataset(get_data(C.TRAIN_DATA_SIZE))
    test_input = Mydataset(get_data(C.TEST_DATA_SIZE))

    train_loader = DataLoader(train_input,
                              batch_size=C.BATCH_SIZE,
                              shuffle=True, num_workers=2)

    test_loader = DataLoader(test_input,
                             batch_size=C.BATCH_SIZE,
                             shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device===>",device)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=C.LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=2, gamma=C.gamma)

    for epoch in range(1, C.EPOCHS + 1):
        if C.RESTORE_MODEL:
            print("模型开始增量训练==>>")
            model = torch.load(C.MODEL_SAVE_PATH)
            mytrain(model, device, train_loader, optimizer, epoch)
            mytest(model, device, test_loader,epoch)
            scheduler.step()

        else:
            mytrain(model, device, train_loader, optimizer, epoch)
            mytest(model, device, test_loader, epoch)
            scheduler.step()


    if C.SAVE_MODEL:
        torch.save(model, C.MODEL_SAVE_PATH)


## 参考了：https://github.com/pytorch/examples/blob/master/mnist/main.py