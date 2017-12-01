import os

BASE_DIR = "/www/data/prinda/statoil/"
PROJECT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../"

EXPERIMENTS_DIR = BASE_DIR + "experiments/"

# Datasets
RAW_TRAINSET = BASE_DIR + "raw/train.json"
RAW_TESTSET = BASE_DIR + "raw/test.json"
# DATA_FOLDER = BASE_DIR + "data01/"
# DATA_FOLDER = BASE_DIR + "data02/"  # band_1, band_2 as np.ndarray
# DATA_FOLDER = BASE_DIR + "data03/"  # trainset expansion - flip
# DATA_FOLDER = BASE_DIR + "data04/"  # trainset expansion - flip, rotate 45degrees
# DATA_FOLDER = BASE_DIR + "data05/"  # traindev also augmented
DATA_FOLDER = BASE_DIR + "data06/"  # dev also also augmented
# DATA_FOLDER = BASE_DIR + "data06_299_299_3/"  # reshaped images to 299x299x3
DATA_FOLDER = BASE_DIR + "data07/"  # normal size, dev augmented, 90°
DATASETS = {
    "train": DATA_FOLDER + "train.pkl",
    "dev": DATA_FOLDER + "dev.pkl",
    "traindev": DATA_FOLDER + "traindev.pkl",
    "test": DATA_FOLDER + "test.pkl",
    "gentrain": DATA_FOLDER + "gentrain.pkl",
    "gendev": DATA_FOLDER + "gendev.pkl",

    "train_short": DATA_FOLDER + "train_short.pkl",
    "dev_short": DATA_FOLDER + "dev_short.pkl",
    "traindev_short": DATA_FOLDER + "traindev_short.pkl",
    "test_short": DATA_FOLDER + "test_short.pkl",
    "gentrain_short": DATA_FOLDER + "gentrain_short.pkl",
    "gendev_short": DATA_FOLDER + "gendev_short.pkl",
}

# Data
IMG_SIZE = 75
NUM_CHANNELS = 3
NUM_LABELS = 1

IMG_SIZE_TL = 299
NUM_CHANNELS_TL = 3

# Models
CONF_DEFAULT = PROJECT_DIR + "conf/default.json"
CONF_SIMPLE_CNN = PROJECT_DIR + "conf/simple_cnn.json"
CONF_SIMPLE_CNN_HYPERTUNER = PROJECT_DIR + "conf/simple_cnn_hypertuner.json"
CONF_CNN_PRETRAIN = PROJECT_DIR + "conf/cnn_pretrain.json"
CONF_CNN_TL = PROJECT_DIR + "conf/cnn_tl.json"
CONF_CNN_INCEPTION = PROJECT_DIR + "conf/cnn_inception.json"


