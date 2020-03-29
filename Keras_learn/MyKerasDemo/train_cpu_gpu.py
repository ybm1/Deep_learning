
import tensorflow as tf

import tensorflow.keras.backend as K
from multi_input_data import get_data
import Const_k as C
from model import keras_model,compile_model


def train_keras():
    train_next_element = get_data(C.TRAIN_RECODER_PATH,C.EPOCHS)
    test_next_element = get_data(C.TEST_RECODER_PATH,C.EPOCHS)

    mymodel = keras_model()
    compile_model(mymodel)
    # 单机多GPU时只需要加上这一行
    # mymodel = tf.keras.utils.multi_gpu_model(mymodel, gpus=4)  # 指定GPU个数

    callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss'),
        # Write TensorBoard logs to `./logs` directory
        tf.keras.callbacks.TensorBoard(log_dir='./logs')
        #tf.keras.callbacks.ModelCheckpoint(C.CLASS_MODEL_SAVE_PATH,
        #                                   monitor='val_loss', verbose=1,
        #                                   save_best_only=True,
        #                                    mode='auto')
        ]

    X = [train_next_element[0],train_next_element[1],
         train_next_element[2]]
    y = train_next_element[3]

    test_X = [test_next_element[0],test_next_element[1],
         test_next_element[2]]
    test_y = test_next_element[3]
    """
    注意，tf.keras可以直接接受由tf.data的迭代器，但是此时要指定steps_per_epoch参数
    该参数为每个epoch训练的步数，即训练数据量/batch_size，每个batch是一个step
    可以配合tf.data中的repeat方法进行使用，多个epoch就是对数据进行多轮使用
    
    """
    if C.RESTORE_MODEL:
        print("开始增量训练===>>>")
        # Recreate the exact same model purely from the file:
        mymodel = tf.keras.models.load_model(C.REGTRSSION_MODEL_SAVE_PATH)
        compile_model(mymodel)
    steps_per_epoch = int(C.TRAIN_DATA_SIZE/C.BATCH_SIZE)
    valid_steps = int(C.TEST_DATA_SIZE/C.BATCH_SIZE)

    mymodel.fit(X,y,epochs=C.EPOCHS,
                steps_per_epoch=steps_per_epoch,
                validation_data=[test_X, test_y],
                validation_steps=valid_steps,
                callbacks = callbacks)

    print("开始保存模型===>>>")
    mymodel.save(C.REGTRSSION_MODEL_SAVE_PATH)

    print("All finished!!! ")





if __name__ == '__main__':
    train_keras()






































