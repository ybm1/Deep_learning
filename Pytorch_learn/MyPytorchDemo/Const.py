

## 训练相关常数
BATCH_SIZE = 36

LEARNING_RATE = 10**(-2)

EPOCHS = 20

LOG_INTERVAL = 3

SAVE_MODEL = True

gamma = 0.1

RESTORE_MODEL = False


PATH_to_log_dir = "./logs"
CLASS_MODEL_SAVE_PATH = "./model_save/My_class_demo.pkl"
REGTRSSION_MODEL_SAVE_PATH = "./model_save/My_regression_demo.pkl"
## 输入相关
TRAIN_DATA_SIZE = 2000

TEST_DATA_SIZE = 500

SAMPLE_SIZE = TRAIN_DATA_SIZE + TEST_DATA_SIZE


## LSTM
TIME_STEP  = 8
INPUT_SIZE = 10
HIDDEN_SIZE = 20
NUM_LAYERS = 2


## FC

OUT_FEATURES = 10

