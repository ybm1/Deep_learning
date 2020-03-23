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


def mytrain(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(train_loader):
        p1 = sample_batched["features"]["p1"]
        p2 = sample_batched["features"]["p2"]
        p3 = sample_batched["features"]["p3"]

        target = sample_batched["label"]

        #p1,p2,p3,target = p1.to(device),p2.to(device),p3.to(device),target.to(device)
        ## gpu时可以直接.cuda() 把数据放在GPU上
        p1, p2, p3, target = p1.cuda(), p2.cuda(), p3.cuda(), target.cuda()

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
            print('Train Epoch: {} Batch_idx :{} \tLoss: {:.8f}'.format(
                epoch, batch_idx , loss.item()))



def mytest(model, test_loader,epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            p1 = data["features"]["p1"]
            p2 = data["features"]["p2"]
            p3 = data["features"]["p3"]

            target = data["label"]

            #p1, p2, p3, target = p1.to(device), p2.to(device), p3.to(device), target.to(device)
            ## gpu时可以直接.cuda() 把数据放在GPU上

            p1, p2, p3, target = p1.cuda(), p2.cuda(), p3.cuda(), target.cuda()

            output = model(p1,p2,p3)
            test_loss += F.mse_loss(input = output, target = target, reduction='sum').item()  # sum up batch loss


    test_loss /= len(test_loader.dataset)
    writer.add_scalar('Epoch test MSE',
                      test_loss,epoch)

    print('\nTest set:Epoch: {}  Average MSE loss: {:.8f}'.format(epoch,test_loss))




if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
    print("device counts===>", torch.cuda.device_count())
    model = Net()
    #if torch.cuda.device_count() > 1:
    device_ids = ["0"] # 在cloab上测试，device_ids需要是字符列表形式
    model = torch.nn.DataParallel(model,device_ids = device_ids).cuda()

    # 经过在colab上的测试，只需要上面这一行就可以，通过device_ids的不同，来指定是单个GPU
    # 还是多个GPU，即使是单个GPU也可以直接跑(colab测试可以成功)，多GPU还没测，不过问题不大

    # model = model.module  # 有博客说多gpu时要加上这行，见下面
    # https://blog.csdn.net/daydayjump/article/details/81158777

    #model = model.to(device)  这行和model.cuda()效果应该一样

    optimizer = optim.Adam(model.parameters(), lr=C.LEARNING_RATE)


    scheduler = StepLR(optimizer, step_size=2, gamma=C.gamma)

    for epoch in range(1, C.EPOCHS + 1):
        if C.RESTORE_MODEL:
            print("模型开始增量训练==>>")
            model = torch.load(C.REGTRSSION_MODEL_SAVE_PATH)
            mytrain(model, train_loader, optimizer, epoch)

            mytest(model, test_loader,epoch)
            scheduler.step()

        else:
            mytrain(model,  train_loader, optimizer, epoch)
            mytest(model, test_loader, epoch)
            scheduler.step()


    if C.SAVE_MODEL:
        torch.save(model, C.REGTRSSION_MODEL_SAVE_PATH)


## 参考了：https://github.com/pytorch/examples/blob/master/mnist/main.py