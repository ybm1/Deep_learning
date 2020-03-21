# Deep_learning
My DL study.

Tf和Pytorch的学习代码整理.

分别用Pytorch和Tf实现了一个Demo，该Demo支持下面的功能：

* 数据输入，模型定义，训练的模块化
* 数据支持多输入的情况，这里的每一个样本的特征包括三部分，
第一部分是3维tensor(模拟图片)，第二部分是二维tensor(模拟句子)，第三部分是一维tensor(模拟一般的ML的特征)
* 支持单机多卡训练
* 支持模型的保存和增量训练
* 支持tensorboard

目前已经用Pytorch写完，完整的项目代码在MyDemo中。

Tf的正在进行。
