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


from sklearn.metrics import *


## 本代码可以对cpu或者单个gpu进行使用

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

        p1,p2,p3,target = p1.to(device,dtype = torch.float32),\
                          p2.to(device,dtype = torch.float32),\
                          p3.to(device,dtype = torch.float32),target.to(device)

        optimizer.zero_grad()
        output = model(p1, p2, p3)

        # 如果是分类，就用CrossEntropyLoss,用这个不用转one-hot 它会自动转
        # 对于CrossEntropyLoss，其target的size应该是torch.Size([batch_size])
        # 其output的size为torch.Size([batch_size, 类别数])
        #print(target.size())
        #print(output.size())
        Cross_Entropy_loss = torch.nn.CrossEntropyLoss()
        loss = Cross_Entropy_loss(output, target)

        loss.backward()

        ## 计算分类的各种指标
        output_cate = torch.argmax(output,dim=1)

        y_true = target.detach().numpy()
        y_pred = output_cate.detach().numpy()
        y_score_max = torch.max(output,dim=1)
        #print(y_score_max[0])
        y_score = y_score_max[0].detach().numpy()


        batch_accuracy = accuracy_score(y_true,y_pred )
        batch_recall = recall_score(y_true,y_pred)
        batch_precision = precision_score(y_true,y_pred)
        batch_auc = roc_auc_score(y_true,y_score)


        optimizer.step()


        writer.add_scalar('training Batch Cross Entropy loss',
                          loss.item(),
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('training Batch Accuracy',
                          batch_accuracy,
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('training Batch Recall',
                          batch_recall,
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('training Batch Precision',
                          batch_precision,
                          epoch * len(train_loader) + batch_idx)
        writer.add_scalar('training Batch AUC',
                          batch_auc,
                          epoch * len(train_loader) + batch_idx)
        if batch_idx % C.LOG_INTERVAL == 0:
            print('Train Epoch: {} Batch_idx :{} \t Batch Cross Entropy loss: {:.8f}'.format(
                epoch, batch_idx, loss.item()))
            print('Train Epoch: {} Batch_idx :{} \tBatch Accuracy: {:.8f}'.format(
                epoch, batch_idx, batch_accuracy))
            print('Train Epoch: {} Batch_idx :{} \tBatch Recall: {:.8f}'.format(
                epoch, batch_idx, batch_recall))
            print('Train Epoch: {} Batch_idx :{} \tBatch Precision: {:.8f}'.format(
                epoch, batch_idx, batch_precision))
            print('Train Epoch: {} Batch_idx :{} \tBatch Auc: {:.8f}'.format(
                epoch, batch_idx, batch_auc))




def mytest(model, device, test_loader,epoch):
    model.eval()
    test_loss = 0
    Y_true = []
    Y_pred = []
    Y_score = []
    with torch.no_grad():
        for data in test_loader:
            p1 = data["features"]["p1"]
            p2 = data["features"]["p2"]
            p3 = data["features"]["p3"]

            target = data["label"]

            p1, p2, p3, target = p1.to(device), p2.to(device), p3.to(device), target.to(device)

            output = model(p1, p2, p3)
            # sum up batch loss
            test_loss +=F.cross_entropy(input=output, target= target,reduction='sum').item()

            output_cate = torch.argmax(output, dim=1)
            y_true = list(target.detach().numpy())
            y_pred = list(output_cate.detach().numpy())
            y_score_max = torch.max(output, dim=1)

            y_score = list(y_score_max[0].detach().numpy())

            Y_true += y_true
            Y_pred += y_pred
            Y_score += y_score


        test_loss /= len(test_loader.dataset)
        #print(Y_pred)

        test_epoch_accuracy = accuracy_score(Y_true, Y_pred)
        test_epoch_recall = recall_score(Y_true, Y_pred)
        test_epoch_precision = precision_score(Y_true, Y_pred)
        test_epoch_auc = roc_auc_score(Y_true, Y_score)
        writer.add_scalar('Epoch test Cross Entropy Loss',
                          test_loss, epoch)
        writer.add_scalar('Epoch test Accuracy',
                          test_epoch_accuracy, epoch)
        writer.add_scalar('Epoch test Precision',
                          test_epoch_precision, epoch)
        writer.add_scalar('Epoch test Recall',
                          test_epoch_recall, epoch)
        writer.add_scalar('Epoch test AUC',
                          test_epoch_auc, epoch)
        print('\nTest set:Epoch: {}  Average Cross Entropy Loss: {:.8f}'.format(epoch, test_loss))
        print('\nTest set:Epoch: {}  Accuracy: {:.8f}'.format(epoch, test_epoch_accuracy))
        print('\nTest set:Epoch: {}  Precision: {:.8f}'.format(epoch, test_epoch_precision))
        print('\nTest set:Epoch: {}  Recall: {:.8f}'.format(epoch, test_epoch_recall))
        print('\nTest set:Epoch: {}  AUC: {:.8f}'.format(epoch, test_epoch_auc))




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
            model = torch.load(C.CLASS_MODEL_SAVE_PATH)
            mytrain(model,device ,train_loader, optimizer, epoch)
            mytest(model, device, test_loader,epoch)
            scheduler.step()

        else:
            mytrain(model,device, train_loader, optimizer, epoch)
            mytest(model, device, test_loader, epoch)
            scheduler.step()


    if C.SAVE_MODEL:
        torch.save(model, C.CLASS_MODEL_SAVE_PATH)


## 参考了：https://github.com/pytorch/examples/blob/master/mnist/main.py