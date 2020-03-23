import tensorflow as tf
import numpy as np
import Const as C

# 参考了： https://blog.csdn.net/u014061630/article/details/80728694
# https://www.cnblogs.com/puheng/p/9576521.html
# https://www.jianshu.com/p/72596a8488c3
# https://blog.csdn.net/wiinter_fdd/article/details/72835939
# https://blog.csdn.net/GodWriter/article/details/90200179

sess = tf.Session()

def generate_tfrecoder(sample_size):
    """
    用于产生tfrecoder的模拟数据，这里对每个样本模拟了3部分特征：
    p1: 3维的Tensor特征(模拟图片)
    p2: 2维的Tensor特征(模拟句子)
    p1: 最一般的向量型(ML)特征
    label是样本的标签，分类和回归时有所不同
    :param sample_size: 样本量
    :return: 无返回值，全部数据写入在 C.RECODER_PATH 下
    """
    writer = tf.io.TFRecordWriter(C.RECODER_PATH )
    for i in range(sample_size):
        features = {}

        p1 = np.random.rand(3, 20, 40)
        # 写入张量，类型float，本身是三维张量，另一种方法是转变成字符类型存储，随后再转回原类型
        features['p1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[p1.tostring()]))
        # 存储丢失的形状信息(10,20,3)
        features['p1_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=p1.shape))


        p2 = np.random.rand(8, 10)
        # 写入矩阵，类型float，本身是矩阵，一种方法是将矩阵flatten成list
        features['p2'] = tf.train.Feature(float_list=tf.train.FloatList(value=p2.reshape(-1)))
        # 然而矩阵的形状信息(2,3)会丢失，需要存储形状信息，随后可转回原形状
        features['p2_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=p2.shape))

        # 写入向量，类型float
        p3 = np.random.rand(10)
        features['p3'] = tf.train.Feature(float_list=tf.train.FloatList(value=p3.tolist()))
        features['p3_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=p3.shape))

        # 写入标量，类型float，要变成list
        label  = np.random.rand(1)
        features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=label.tolist()))
        features['label_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label.shape))

        # 将存有所有feature的字典送入tf.train.Features中
        tf_features = tf.train.Features(feature=features)
        # 再将其变成一个样本example
        tf_example = tf.train.Example(features=tf_features)
        # 序列化该样本
        tf_serialized = tf_example.SerializeToString()


        if (i+1) % 50 ==0:
            print("第 {} 个 tf recoder正在写入...".format(i+1))
        # 写入一个序列化的样本
        writer.write(tf_serialized)
        # 由于上面有循环几次，我们就已经写了几个样本
        # 关闭文件
    writer.close()
    print("tf recoder 全部写入完毕！")


generate_tfrecoder(3000)


def parse_function(example_proto):
    # 定义解析的字典
    dics = {
        'label': tf.io.FixedLenFeature([], tf.float32),
        'label_shape': tf.io.FixedLenFeature([], tf.int64),
        'p1': tf.io.FixedLenFeature([], tf.string),
        'p1_shape': tf.io.FixedLenFeature([], tf.int64),
        'p2': tf.io.FixedLenFeature([], tf.float32),
        'p2_shape': tf.io.FixedLenFeature([], tf.int64),
        'p3': tf.io.FixedLenFeature([], tf.float32),
        'p3_shape': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_example = tf.io.parse_single_example(serialized=example_proto, features=dics)

    label = parsed_example["label"]
    label = tf.reshape(label, shape= parsed_example["label_shape"])
    p1 = parsed_example["p1"]
    p1 = tf.reshape(p1, shape= parsed_example["p1_shape"])

    p2 = parsed_example["p2"]
    p2 = tf.reshape(p2, shape= parsed_example["p2_shape"])

    p3 = parsed_example["p3"]
    p3 = tf.reshape(p3, shape= parsed_example["p3_shape"])


    return p1,p2,p3,label


def get_data(filename):
    dataset = tf.data.TFRecordDataset(filenames=[filename])
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle()
    dataset = dataset.batch(C.BATCH_SIZE).repeat(1)

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()


    sess.run(iterator.initializer)
    # Compute for 100 epochs.

    for _ in range(100):
        while True:
            try:
                print(sess.run(next_element))
            except tf.errors.OutOfRangeError:
                break








if __name__ == '__main__':
    #generate_tfrecoder(2000)


    get_data(C.RECODER_PATH)


















