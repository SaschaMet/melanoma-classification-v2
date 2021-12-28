import os
import random
import numpy as np
import tensorflow as tf

from hyperparameters.set_hyperparameters import set_hyperparameters

SEED = 42


def main():
    HYPERPARAMETERS = {}
    HYPERPARAMETERS.update({"VALIDATION_SIZE": 0.15})
    HYPERPARAMETERS.update({"IMG_SHAPE": 256})
    HYPERPARAMETERS.update({"BATCH_SIZE": 32})
    HYPERPARAMETERS.update({"REPLICAS": 1})
    HYPERPARAMETERS.update({"SEED": SEED})

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    set_hyperparameters(HYPERPARAMETERS)


main()
