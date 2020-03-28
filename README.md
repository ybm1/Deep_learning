# Deep_learning
My DL study.

`Pytorch`、`Tf`和`tf.keras`的学习代码整理.

分别用`Pytorch`、`Tf(1.x)`和`tf.keras`实现了一个Demo，该Demo虽然小，但是五脏俱全，
它有下面的特点：

* 均支持数据输入，模型定义，训练的模块化.
* 均支持数据多输入和批量训练.
>这里的多输入指的是，每一个样本的都特征包括三部分:

>第一部分是三维`tensor`(模拟图片)

>第二部分是二维`tensor`(模拟句子)

>第三部分是一维`tensor`(模拟一般的ML的特征)

>在`Pytorch`的例子中，该部分使用了`Dataset`和`DataLoader`的API.

>在`Tf`的例子中，该部分使用了`tf.data.Dataset`的`API`，支持`tfrecoder`的写入生成，读取和解析.

>在`tf.keras`的例子中，该部分可以无缝衔接的使用`Tf`中的`tf.data.Dataset`的`API`.

* 均支持单机多卡训练.
* 均支持模型的保存和加载(增量训练).
>在`Tf`的例子中，模型的定义、单机多卡训练、保存和加载中基本使用的
是`Tf`的**低级**`API`,而高级`API`在`tf.keras`中使用.

* 均支持`tensorboard`可视化.




`Pytorch`的完整的项目代码在`Pytorch_learn/MyPytorchDemo`中.

`Tensorflow`的完整的项目代码在`Tf_learn/MyTfDemo`中.

`tf.keras`的完整的项目代码在`Keras_learn/MyKerasDemo`中.

参考的一些资料和博客在`参考资料`中.