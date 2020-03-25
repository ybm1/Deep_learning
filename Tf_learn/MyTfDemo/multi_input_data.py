import tensorflow as tf
import numpy as np
import Const as C

# 参考了： https://blog.csdn.net/u014061630/article/details/80728694
# https://www.cnblogs.com/puheng/p/9576521.html
# https://www.jianshu.com/p/72596a8488c3
# https://blog.csdn.net/wiinter_fdd/article/details/72835939
# https://blog.csdn.net/GodWriter/article/details/90200179

sess = tf.Session()

def generate_tfrecoder(sample_size,path):
    """
    用于产生tfrecoder的模拟数据，这里对每个样本模拟了3部分特征：
    p1: 3维的Tensor特征(模拟图片)
    p2: 2维的Tensor特征(模拟句子)
    p1: 最一般的向量型(ML)特征
    label是样本的标签，分类和回归时有所不同
    :param sample_size: 样本量
    :return: 无返回值，全部数据写入在 path 下
    """
    writer = tf.io.TFRecordWriter(path )
    for i in range(sample_size):
        features = {}

        p1 = np.random.rand(20, 40,3)
        # 写入张量，类型float，本身是三维张量 一种方法是将矩阵flatten成list
        # 另一种方法是转变成字符类型存储，随后再转回原类型 这里用第一种第二种转字符的不太好操作
        #features['p1'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[p1.tostring()]))

        features['p1'] = tf.train.Feature(float_list=tf.train.FloatList(value=p1.reshape(-1)))
        # 存储丢失的形状信息(10,20,3)
        features['p1_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=p1.shape))

        #print(features["p1_shape"])
        p2 = np.random.rand(8, 10)
        # 写入矩阵，类型float，本身是矩阵，一种方法是将矩阵flatten成list
        features['p2'] = tf.train.Feature(float_list=tf.train.FloatList(value=p2.reshape(-1)))
        # 然而矩阵的形状信息(2,3)会丢失，需要存储形状信息，随后可转回原形状
        features['p2_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=p2.shape))
        #print(features["p2_shape"])
        # 写入向量，类型float
        p3 = np.random.rand(10)
        features['p3'] = tf.train.Feature(float_list=tf.train.FloatList(value=p3.tolist()))
        features['p3_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=p3.shape))
        #print(features["p3_shape"])
        # 写入标量，类型float，要变成list
        label  = np.random.rand(1)
        features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=label.tolist()))
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
        # 上面有循环几次，我们就已经写了几个样本
        # 关闭文件
    writer.close()
    print("tf recoder 全部写入完毕！")




def parse_function(example_proto):
    # 定义解析的字典
    dics = {

        'p1': tf.io.VarLenFeature(dtype = tf.float32),

        'p1_shape': tf.io.FixedLenFeature(shape=(3,), dtype = tf.int64),

        'p2': tf.io.VarLenFeature( dtype = tf.float32),
        'p2_shape': tf.io.FixedLenFeature(shape=(2,), dtype =tf.int64),

        'p3': tf.io.FixedLenFeature(shape= (10), dtype = tf.float32),

        'p3_shape': tf.io.FixedLenFeature([], tf.int64),

        'label': tf.io.FixedLenFeature(shape=(1), dtype=tf.float32)
    }
    parsed_example = tf.io.parse_single_example(serialized=example_proto, features=dics)
    """
    一定注意！！！
    
    对于二维和三维的tensor，解析的时候要用不定长的tf.io.VarLenFeature
    并且要把稀疏的转化为稠密的，
    然后再做reshape
    另外，解析的时候输入的type要和输出的type保持一致
    """

    p1 = tf.sparse_tensor_to_dense(parsed_example['p1'])
    p1 = tf.reshape(p1,(20, 40,3))
    p2 = tf.sparse_tensor_to_dense(parsed_example['p2'])
    p2 = tf.reshape(p2,(8, 10))
    p3 = parsed_example["p3"]

    label = parsed_example["label"]

    return p1,p2,p3,label


def get_data(filename):
    dataset = tf.data.TFRecordDataset(filenames=[filename])
    #print("读取tfrecoder 成功...")
    dataset = dataset.map(parse_function)
    #dataset = dataset.shuffle(buffer_size=C.TRAIN_DATA_SIZE)
    dataset = dataset.batch(C.BATCH_SIZE)
    # 用迭代器进行batch的读取
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # 返回next_element后，反复的run它，就可以得到不同的batch
    # 这就是输入数据和训练模型时的连接点
    return next_element








if __name__ == '__main__':
    #generate_tfrecoder(C.TRAIN_DATA_SIZE,C.TRAIN_RECODER_PATH)
    #generate_tfrecoder(C.TEST_DATA_SIZE, C.TEST_RECODER_PATH)


    # Compute for epochs.
    for i in range(C.EPOCHS):
        train_next_element = get_data(C.TRAIN_RECODER_PATH)
        print("epoch {}".format(i+1))
        while True:
            try:
                p1, p2, p3, label = sess.run([train_next_element[0],
                                              train_next_element[1],
                                              train_next_element[2],
                                              train_next_element[3]])
                print(p1.shape)
                print(p2.shape)
                print(p3.shape)
                print(label.shape)

            except tf.errors.OutOfRangeError:
                break


















