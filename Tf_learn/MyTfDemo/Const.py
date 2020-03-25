

## 训练相关常数
BATCH_SIZE = 36

LEARNING_RATE = 10**(-2)

EPOCHS = 5

LOG_INTERVAL = 3

SAVE_MODEL = True

gamma = 0.1

RESTORE_MODEL = False

DROPOUT_RATE = 0.3

PATH_to_log_dir = "./logs"
CLASS_MODEL_SAVE_PATH = "./model_save/My_class_demo.pkl"
REGTRSSION_MODEL_SAVE_PATH = "./model_save/My_regression_demo.pkl"
## 输入相关
TRAIN_DATA_SIZE = 1000

TEST_DATA_SIZE = 300

SAMPLE_SIZE = TRAIN_DATA_SIZE + TEST_DATA_SIZE


TRAIN_RECODER_PATH = "./Tfdata/train.tfrecords"
TEST_RECODER_PATH = "./Tfdata/test.tfrecords"


## LSTM
TIME_STEP  = 8
INPUT_SIZE = 10
HIDDEN_SIZE = 20
NUM_LAYERS = 2


## FC

OUT_FEATURES = 10









