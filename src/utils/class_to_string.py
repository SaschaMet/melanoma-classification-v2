from data_preparation.create_datasets import map_classes
from hyperparameters.get_hyperparameter import get_hyperparameter


def class_to_string(x):
    CLASSES = get_hyperparameter("CLASSES")
    label_string = CLASSES.get(str(x))
    return label_string
