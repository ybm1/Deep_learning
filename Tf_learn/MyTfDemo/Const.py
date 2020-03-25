

## 训练相关常数
BATCH_SIZE = 36

LEARNING_RATE = 10**(-2)

EPOCHS = 3

LOG_INTERVAL = 3

SAVE_MODEL = True

gamma = 0.1

RESTORE_MODEL = False

DROPOUT_RATE = 0.3


TRAIN_PATH_to_log_dir = "./logs/train"
TEST_PATH_to_log_dir = "./logs/test"


CLASS_MODEL_SAVE_PATH = "./model_save/My_class_model.ckpt"
REGTRSSION_MODEL_SAVE_PATH = "./model_save/My_reg_model.ckpt"
## 输入相关
TRAIN_DATA_SIZE = 1000

TEST_DATA_SIZE = 300

SAMPLE_SIZE = TRAIN_DATA_SIZE + TEST_DATA_SIZE

MAX_STEP = int(TRAIN_DATA_SIZE/BATCH_SIZE)
TRAIN_RECODER_PATH = "./Tfdata/train.tfrecords"
TEST_RECODER_PATH = "./Tfdata/test.tfrecords"


## LSTM
TIME_STEP  = 8
INPUT_SIZE = 10
HIDDEN_SIZE = 20
NUM_LAYERS = 2


## FC

OUT_FEATURES = 10









