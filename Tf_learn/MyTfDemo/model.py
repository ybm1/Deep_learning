import tensorflow as tf
import Const as C
from multi_input_data import get_data

# 定义权重来生成全连接层
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 定义偏置来生成全连接层
def bias_variable(shape,relu=True):
    if relu:
        # 卷积中用relu激活函数，所以用一个很小的正偏置较好
        initial = tf.constant(0.1, shape=shape)
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义卷积层
def conv2d1(input,W):

    return tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME')

def conv2d2(input,W):

    return tf.nn.conv2d(input,W, strides=[1,1,1,1], padding='SAME')


# pooling 层
def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')


# lstm层
def get_lstm(n_hidden, keep_prob,name):
    lstm = tf.nn.rnn_cell.LSTMCell(n_hidden,name = name)
    dropped = tf.nn.rnn_cell.DropoutWrapper(lstm, keep_prob,name= name)
    return dropped

def MyNet(p1,p2,p3):


    with tf.variable_scope("p1_3d_tensor_process"):
        # 可以用placeholder来看shape的变化
        #p1 = tf.placeholder(tf.float32, [None, 20, 40, 3])
        #p1 = tf.reshape(p1,[None,20, 40, 3])
        # 第一层卷积：5×5×1卷积核6个 [5，5，3，6]
        W_conv1 = weight_variable([5, 5, 3, 6])
        b_conv1 = bias_variable([6])
        h_conv1 = tf.nn.relu(conv2d1(p1, W_conv1) + b_conv1)
        # 第一个pooling 层
        h_pool1 = max_pool_2x2(h_conv1)
        # 第二层卷积：3×3×6卷积核12个 [3，3，6，12]
        W_conv2 = weight_variable([3, 3, 6, 12])
        b_conv2 = bias_variable([12])
        h_conv2 = tf.nn.relu(conv2d2(h_pool1, W_conv2) + b_conv2)

        # 第二个pooling 层,输出(?, 20, 40, 12)
        h_pool2 = max_pool_2x2(h_conv2)

        #print(p1,h_conv1,h_pool1,h_conv2,h_pool2)

        # flatten
        h_pool2_flat = tf.reshape(h_pool2, [-1, 20*40*12])

        # fc1
        W_fc1 = weight_variable([20*40*12, 10])
        b_fc1 = bias_variable([10])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
        ##rate1 = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, rate=C.DROPOUT_RATE)
        # 输出层
        W_fc2 = weight_variable([10, 10])
        b_fc2 = bias_variable([10])
        p1_output = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        #print(p1_output)

    with tf.variable_scope("p2_2d_tensor_process",reuse=tf.AUTO_REUSE):
        #p2 = tf.placeholder(tf.float32,[None,8, 10])
        #p2 = tf.reshape(p2, [None,8, 10])
        # Define lstm cells with tensorflo
        # Forward direction cell
        rate2 = tf.placeholder(tf.float32)
        with tf.variable_scope("lstm1"):
            lstm_fw_cell = get_lstm(C.HIDDEN_SIZE,1-C.DROPOUT_RATE,name= "lstm_fw")
        # Backward direction cell
        with tf.variable_scope("lstm2"):
            lstm_bw_cell = get_lstm(C.HIDDEN_SIZE,1-C.DROPOUT_RATE,name= "lstm_bw")

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                          cell_bw=lstm_bw_cell,
                                                          dtype=tf.float32,
                                                          inputs=p2)

        output_fw, output_bw = outputs
        states_fw, states_bw = states
        #print(output_fw,output_bw)
        #print(states_fw,states_bw)
        lstm_output = tf.concat([output_fw, output_bw], 2)
        #print(lstm_output.shape)
        #lstm_output = tf.reshape(lstm_output, [None, 8*40])
        lstm_output = tf.layers.flatten(lstm_output)
        #print(lstm_output.shape)
        # 输出层
        W_fc_lstm = weight_variable([8*40, 10])
        b_fc_lstm = bias_variable([10],relu=False)

        p2_output = tf.nn.sigmoid(tf.matmul(lstm_output, W_fc_lstm) + b_fc_lstm)
        #print(p2_output)

    with tf.variable_scope("p3_1d_tensor_process"):
        #p3 = tf.placeholder(tf.float32,[None,10])
        #p3 = tf.reshape(p3, [None,10])
        W_fc_p3 = weight_variable([10, 10])
        b_fc_p3 = bias_variable([10], relu=False)
        output = tf.nn.sigmoid(tf.matmul(p3, W_fc_p3) + b_fc_p3)

        W_fc_p3_ = weight_variable([10, 10])
        b_fc_p3_ = bias_variable([10], relu=False)
        p3_output = tf.nn.sigmoid(tf.matmul(output, W_fc_p3_) + b_fc_p3_)
    #print(p1_output,p2_output,p3_output)

    all_concat = tf.concat([p1_output,p2_output,p3_output],1)
    W_fc_all = weight_variable([3*10, 1])
    b_fc_all = bias_variable([1], relu=False)

    y_pred = tf.nn.sigmoid(tf.matmul(all_concat, W_fc_all) + b_fc_all)
    #print(y_pred)
    return y_pred








if __name__ == '__main__':
    # Compute for epochs.
    for _ in range(C.EPOCHS):
        next_element = get_data(C.TRAIN_RECODER_PATH)
        while True:
            try:
                p1, p2, p3 = next_element[0],next_element[1],next_element[2]
                print(p1,p2,p3)
                MyNet(p1,p2,p3)
            except tf.errors.OutOfRangeError:
                break

























