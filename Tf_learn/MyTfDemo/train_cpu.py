import tensorflow as tf
from multi_input_data import get_data
import Const as C
import model as m
# 设置GPU按需增长
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

def train_on_cpu():

    p1 = tf.placeholder(tf.float32, [C.BATCH_SIZE, 20, 40, 3])
    p2 = tf.placeholder(tf.float32, [C.BATCH_SIZE, 8, 10])
    p3 = tf.placeholder(tf.float32, [C.BATCH_SIZE, 10])
    label = tf.placeholder(tf.float32,[C.BATCH_SIZE,1])
    y_pred = m.MyNet(p1,p2,p3)

    #mse = tf.reduce_sum((label -y_pred)**2)
    mse = tf.losses.mean_squared_error(label, y_pred)
    #mse = tf.nn.l2_loss(label,y_pred)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(mse)

    sess.run(tf.global_variables_initializer())

    # Compute for epochs.
    for i in range(C.EPOCHS):
        j = 0
        while True:
            try:
                j+=1
                next_element = get_data(C.RECODER_PATH)
                p1_run, p2_run, p3_run, label_run = sess.run([next_element[0],
                                              next_element[1],
                                              next_element[2],
                                              next_element[3]])
                # 注意mse_loss不要和右边的mse重名，要不然报错
                mse_loss, _ = sess.run([mse,train_step],feed_dict={
                    p1:p1_run,
                    p2:p2_run,
                    p3:p3_run,
                    label:label_run
                })
                print("Epoch {} Batch {} : MSE ==> {}".format(i+1,j,mse_loss))
            except tf.errors.OutOfRangeError:
                break




if __name__ == '__main__':
    train_on_cpu()











