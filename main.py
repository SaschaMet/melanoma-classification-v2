import os
import random
import numpy as np
import tensorflow as tf

from hyperparameters.set_hyperparameters import set_hyperparameters

SEED = 42
CLASSES = {
    "0": 'unknown',
    "1": 'nevus',
    "2": 'melanoma',
    "3": 'seborrheic keratosis',
    "4": 'lentigo NOS',
    "5": 'lichenoid keratosis',
    "6": 'solar lentigo',
    "7": 'cafe-au-lait macule',
    "8": 'atypical melanocytic proliferation',
    "11": 'basal cell carcinoma',
    "12": 'actinic keratosis',
    "14": 'dermatofibroma',
    "15": 'vascular lesion',
    "16": 'squamous cell carcinoma',
}

CLASS_MAPPING = {
    "-1": "0",
    "9": '2',
    "10": '1',
    "13": '6',
    "17": '0',
}


def main():
    HYPERPARAMETERS = {}
    HYPERPARAMETERS.update({"VALIDATION_SIZE": 0.15})
    HYPERPARAMETERS.update({"IMG_SHAPE": 256})
    HYPERPARAMETERS.update({"BATCH_SIZE": 32})
    HYPERPARAMETERS.update({"REPLICAS": 1})
    HYPERPARAMETERS.update({"CLASSES": CLASSES})
    HYPERPARAMETERS.update({"CLASS_MAPPING": CLASS_MAPPING})

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    set_hyperparameters(HYPERPARAMETERS)


main()
