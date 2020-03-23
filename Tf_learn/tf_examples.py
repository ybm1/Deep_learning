import tensorflow as tf
import pandas as pd

import numpy as np

dataset1 = tf.data.Dataset.from_tensor_slices(tf.random.uniform([4, 10]))
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

dataset4 = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random.uniform([4]),
    "b": tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset4.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset4.output_shapes)  # ==> "{'a': (), 'b': (100,)}"



writer = tf.io.TFRecordWriter("./Tfdata")



# 这里我们将会写3个样本，每个样本里有4个feature：标量，向量，矩阵，张量
for i in range(3):
    # 创建字典
    features = {}
    # 写入标量，类型Int64，由于是标量，所以"value=[scalars[i]]" 变成list
    scalar =  np.random.randint(10,size=1)
    features['scalar'] = tf.train.Feature(int64_list=tf.train.Int64List(value=scalar.tolist()))

    # 写入向量，类型float，本身就是list，所以"value=vectors[i]"没有中括号
    vector = np.random.rand(10)
    features['vector'] = tf.train.Feature(float_list=tf.train.FloatList(value=vector.tolist()))

    tensor_2d = np.random.rand(10,20)
    # 写入矩阵，类型float，本身是矩阵，一种方法是将矩阵flatten成list
    features['matrix'] = tf.train.Feature(float_list=tf.train.FloatList(value=tensor_2d.reshape(-1)))
    # 然而矩阵的形状信息(2,3)会丢失，需要存储形状信息，随后可转回原形状
    features['matrix_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=tensor_2d.shape))

    tensor_3d = np.random.rand(10,20,3)
    # 写入张量，类型float，本身是三维张量，另一种方法是转变成字符类型存储，随后再转回原类型
    features['tensor'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_3d.tostring()]))
    # 存储丢失的形状信息(10,20,3)
    features['tensor_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=tensor_3d.shape))

    # 将存有所有feature的字典送入tf.train.Features中
    tf_features = tf.train.Features(feature=features)
    # 再将其变成一个样本example
    tf_example = tf.train.Example(features=tf_features)
    # 序列化该样本
    tf_serialized = tf_example.SerializeToString()

    # 写入一个序列化的样本
    writer.write(tf_serialized)
    # 由于上面有循环3次，所以到此我们已经写了3个样本
    # 关闭文件
writer.close()