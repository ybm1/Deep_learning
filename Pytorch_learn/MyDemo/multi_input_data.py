
import torch
import numpy as np
import Const as C

from torch.utils.data import Dataset, DataLoader


# 参考了： https://www.jianshu.com/p/6e22d21c84be

# 通过模拟产生多输入的数据
# 所谓的多输入，即一条样本，可能有很多部分的特征，
# 比如第一部分是个图片(三维Tensor)，第二部分是模拟nlp中的一个句子的形式
# 即二维的Tensor，第三部分模拟一般的DataFrame格式，即是一维Tensor


def get_data(sample_size):
    data = []
    for i in range(sample_size):
        h = {}
        # 模拟图片的形式
        n1 = np.random.random(2400).reshape((12, 20, 10))
        p1 = torch.from_numpy(n1)

        # 模拟nlp中的一个句子的形式，二维Tensor
        # 句子长度为8，每个词向量的长度为10
        n2 = np.random.random(80).reshape((8, 10))
        p2 = torch.from_numpy(n2)

        # 模拟一般的DataFrame格式,Tensor长度为10
        n3 = np.random.random(10).reshape((1, -1))
        p3 = torch.from_numpy(n3)
        # 把特征部分放在字典里，这就是一个样本的所有的特征
        features = {"p1": p1, "p2": p2, "p3": p3}
        # 生成该条样本的label
        label = np.random.random(1)

        # 分别储存features和label
        h["features"] = features
        h["label"] = label
        data.append(h)

    return data



class Mydataset(Dataset):
    """
    定义自己的Dateset
    必须重写__len__方法和__getitem__方法
    """

    def __init__(self,all_data):
        self.data = all_data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, item):

        sample = self.data[item]
        return sample



if __name__ == '__main__':

    train_input = Mydataset(get_data(C.TRAIN_DATA_SIZE))
    test_input = Mydataset(get_data(C.TEST_DATA_SIZE))

    for i in range(len(train_input)):
        sample = train_input[i]

        print(i, sample['features'], sample['label'])
        if i ==1:
            break

    train_loader = DataLoader(train_input,
                            batch_size=C.BATCH_SIZE,
                            shuffle=True, num_workers=2)

    test_loader = DataLoader(test_input,
                            batch_size=C.BATCH_SIZE,
                            shuffle=True, num_workers=2)

    for i_batch, sample_batched in enumerate(train_loader):

        print(i_batch, sample_batched['features'],
              sample_batched['label'])
        if i ==2:
            break














