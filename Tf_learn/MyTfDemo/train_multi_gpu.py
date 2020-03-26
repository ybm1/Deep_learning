import tensorflow as tf
import Const_tf as C
from multi_input_data import get_data,get_data_gpu
import model as m
import warnings
warnings.filterwarnings("ignore")

"""
单GPU的和单CPU的代码几乎没有区别，这里是多GPU的训练代码
采用最简单的数据并行方式
主要参考了：
https://blog.csdn.net/minstyrain/article/details/80986397

https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/6_MultiGPU/multigpu_cnn.ipynb

https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/6_MultiGPU/multigpu_basics.ipynb

"""


def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


num_gpus = len(get_available_gpus())
print("Available GPU Number :"+str(num_gpus))



def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']


def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "/" + ps_device
        else:
            return device

    return _assign



# 设置GPU按需增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

train_writer = tf.summary.FileWriter(C.TRAIN_PATH_to_log_dir, sess.graph)
test_writer = tf.summary.FileWriter(C.TEST_PATH_to_log_dir)


# tensorboard --logdir=/Users/biqixuan/PycharmProjects/Deep_learning/Tf_learn/MyTfDemo/logs/train



def get_loss(train_next_element,p1,p2,p3,label,mse,train_opti,all_steps,writer,
             loss_summary,loss_hist):
    p1_run, p2_run, p3_run, label_run = sess.run([train_next_element[0],
                                                  train_next_element[1],
                                                  train_next_element[2],
                                                  train_next_element[3]])
    # 注意mse_loss不要和右边的mse重名，要不然报错
    mse_loss, _, mse_summ,mse_hist= sess.run([mse, train_opti,loss_summary,loss_hist],
                                             feed_dict={
                                                p1: p1_run,
                                                p2: p2_run,
                                                p3: p3_run,
                                                label: label_run
                                                        })

    writer.add_summary(mse_summ, all_steps)
    writer.add_summary(mse_hist,all_steps)
    return mse_loss



def train_on_gpu():
    with tf.device("/cpu:0"):
        global_step = tf.train.get_or_create_global_step()
        tower_grads = []
        p1 = tf.placeholder(tf.float32, [None, 20, 40,3])
        p2 = tf.placeholder(tf.float32, [None, 8, 10])
        p3 = tf.placeholder(tf.float32,[None,10])
        label = tf.placeholder(tf.float32,[None,1])



        train_opti = tf.train.AdamOptimizer(1e-4)

        saver = tf.train.Saver(max_to_keep=10)

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
                    _x1 = p1[i * C.BATCH_SIZE:(i + 1) * C.BATCH_SIZE]
                    _x2 = p2[i * C.BATCH_SIZE:(i + 1) * C.BATCH_SIZE]
                    _x3 = p3[i * C.BATCH_SIZE:(i + 1) * C.BATCH_SIZE]
                    _y = label[i * C.BATCH_SIZE:(i + 1) * C.BATCH_SIZE]
                    y_pred = m.MyNet(_x1, _x2, _x3)

                    tf.get_variable_scope().reuse_variables()
                    mse = tf.losses.mean_squared_error(_y, y_pred)
                    mse_train_summary = tf.summary.scalar('train mse', mse)
                    mse_test_summary = tf.summary.scalar('test mse', mse)
                    mse_train_hist = tf.summary.histogram('train_mse_hist', mse)
                    mse_test_hist = tf.summary.histogram('test_mse_hist', mse)

                    grads = train_opti.compute_gradients(mse)
                    tower_grads.append(grads)

        grads = average_gradients(tower_grads)
        train_opti = train_opti.apply_gradients(grads)


        sess.run(tf.global_variables_initializer())
        # Compute for epochs.
        all_train_steps = 0
        all_test_steps = 0
        if C.RESTORE_MODEL:
            saver.restore(sess, tf.train.latest_checkpoint(C.LOAD_MODEL_PATH))
            # search for checkpoint file
            print("模型开始增量训练====>")
            graph = tf.get_default_graph()
        for i in range(C.EPOCHS):
            train_next_element = get_data_gpu(C.TRAIN_RECODER_PATH,num_gpus*C.BATCH_SIZE)
            test_next_element = get_data(C.TEST_RECODER_PATH)
            train_bacth_per_epoch = 0
            while True:
                try:
                    train_bacth_per_epoch+=1
                    all_train_steps +=1

                    train_mse_loss = get_loss(train_next_element,p1, p2, p3, label,
                                              mse, train_opti,all_train_steps,train_writer,
                                              mse_train_summary,mse_train_hist)

                    print("Train Epoch {} Batch {} All steps {}: "
                          "MSE ==> {}".format(i + 1, train_bacth_per_epoch,
                                              all_train_steps, train_mse_loss))

                    if all_train_steps % C.SAVE_FREQUENCY ==0:
                        print("model is saved at all train step {}".format(all_train_steps))
                        saver.save(sess, C.REGTRSSION_MODEL_SAVE_PATH,global_step=C.MAX_STEP)

                except tf.errors.OutOfRangeError:
                    break

            test_loss = 0
            test_batchs = 0
            while True:
                try:
                    all_test_steps+=1
                    test_mse_loss = get_loss(test_next_element, p1, p2, p3, label,
                                             mse, train_opti,all_test_steps,test_writer,
                                             mse_test_summary,mse_test_hist)
                    test_loss += test_mse_loss
                    test_batchs += 1
                    print("Test batch {} : MSE ==> {}".format(test_batchs, test_mse_loss))
                    tf.summary.histogram('test_mse_loss_steps', test_mse_loss)

                except tf.errors.OutOfRangeError:
                    break
            test_loss_ave = test_loss / test_batchs
            print("Epoch {} : All Test Data Average MSE {}".format(i+1,test_loss_ave))

        test_writer.close()
        train_writer.close()






if __name__ == '__main__':
    num_gpus = len(get_available_gpus())
    print("Available GPU Number :" + str(num_gpus))

    train_on_gpu()





















