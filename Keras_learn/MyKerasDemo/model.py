
import tensorflow as tf
import Const_k as C

def keras_model():

    p1_input = tf.keras.Input(shape = (20, 40, 3),name = "3d_tensor")

    p2_input = tf.keras.Input(shape=(8, 10), name="2d_tensor")

    p3_input = tf.keras.Input(shape=(10,), name="1d_tensor")


    p1 = tf.keras.layers.Conv2D(activation='relu',filters =12,kernel_size=5,strides =1,padding='same')(p1_input)
    p1 = tf.keras.layers.MaxPooling2D(3)(p1)
    p1 = tf.keras.layers.Conv2D(activation='relu',filters =10,kernel_size=3,strides =1,padding='same')(p1)
    p1 = tf.keras.layers.MaxPooling2D(3)(p1)
    p1 = tf.keras.layers.Reshape([2*4*10])(p1)
    p1 = tf.keras.layers.Dense(10, activation='relu')(p1)
    p1 = tf.keras.layers.Dropout(0.5)(p1)
    p1 = tf.keras.layers.Dense(10, activation='relu')(p1)

    p2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM
                                               (units=C.HIDDEN_SIZE,
                                                dropout=0.3,
                                                return_sequences=True,
                                                input_shape=(8,10)))(p2_input)
    p2 = tf.keras.layers.Reshape([8*40])(p2)
    p2 = tf.keras.layers.Dense(10, activation='tanh')(p2)

    p3 = tf.keras.layers.Dense(10, activation='sigmoid')(p3_input)

    pred = tf.keras.layers.Concatenate(axis=1)([p1, p2, p3])

    pred = tf.keras.layers.Dense(1, activation='sigmoid')(pred)

    # build model
    model = tf.keras.Model(inputs=[p1_input,p2_input,p3_input],
                           outputs=pred,name='My_keras_model')
    print(model.summary())
    return model



def compile_model(model):
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=C.LEARNING_RATE,
                                                   beta1=C.beta1,
                                                   beta2=C.beta2,
                                                   epsilon=C.epsilon),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(),
                           tf.keras.metrics.MeanAbsoluteError() ])
    print("compile 成功...")




if __name__ == '__main__':
    model = keras_model()
    compile_model(model)




















