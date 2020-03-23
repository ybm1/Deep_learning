# Deep_learning
My DL study.

`Tf`和`Pytorch`的学习代码整理.

分别用`Pytorch`和`Tf`实现了一个Demo，该Demo有下面的特点：

* 数据输入，模型定义，训练的模块化
* 数据支持多输入的情况，这里的每一个样本的特征包括三部分，
第一部分是三维`tensor`(模拟图片)，第二部分是二维`tensor`(模拟句子)，
第三部分是一维`tensor`(模拟一般的ML的特征)
* 支持批量训练
* 支持单机多卡训练
* 支持模型的保存和加载(增量训练)
* 支持`tensorboard`可视化
* 分别写了回归和分类的例子，回归的例子中用`MSE`和`bias`进行评估，
分类的例子中加了`auc`、`recall`等常用评估指标
* 在`Tf`例子中，分别使用了`Tf`的低阶`API`和高阶`API tf.keras`实现了
模型的定义、单机多卡训练、保存和加载
* 在`Tf`例子中，输入部分使用了`tf.data.Dataset`的`API`，
支持`tfrecoder`的写入，读取和解析


目前已经用`Pytorch`写完，完整的项目代码在`Pytorch_learn/MyDemo`中。

Tf的正在进行。
