# Deep_learning
My DL study.

`Tf`和`Pytorch`的学习代码整理.

分别用`Pytorch`和`Tf(1.x)`实现了一个Demo，该Demo虽然小，但是五脏俱全，
它有下面的特点：

* 支持数据输入，模型定义，训练的模块化
* 支持数据多输入的情况
>这里的每一个样本的特征包括三部分:

>第一部分是三维`tensor`(模拟图片)

>第二部分是二维`tensor`(模拟句子)

>第三部分是一维`tensor`(模拟一般的ML的特征)

* 支持批量训练
* 支持单机多卡训练
* 支持模型的保存和加载(增量训练)
* 支持`tensorboard`可视化
* 另外，在`Tf`例子中，基本使用了`Tf`的**低阶**`API`实现了
模型的定义、单机多卡训练、保存和加载
* 在`Tf`多输入部分使用了`tf.data.Dataset`的`API`，
支持`tfrecoder`的写入生成，读取和解析


`Pytorch`的完整的项目代码在`Pytorch_learn/MyPytorchDemo`中。

`Tensorflow`的完整的项目代码在`Tf_learn/MyTfDemo`中。
