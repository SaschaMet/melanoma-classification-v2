import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hyperparameters.get_hyperparameter import get_hyperparameter


def create_dataframes():
    """Creates and returns training and validation dataframes as well as the target bias

    Returns:
        Tuple: Training DF, validation DF and target bias
    """
    cwd = os.getcwd()
    SEED = get_hyperparameter("SEED")
    VALIDATION_SIZE = get_hyperparameter("VALIDATION_SIZE")

    df_train = pd.read_csv(os.path.join(cwd + '/train.csv'))

    df_train = df_train.sample(frac=1)
    df_train = df_train.sample(frac=1)
    df_train = df_train.sample(frac=1)

    train, val = train_test_split(
        df_train,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=df_train[['target']])

    # Get the class weights and the inital bias
    malignant_cases = df_train['target'].value_counts()[1]
    benign_cases = df_train['target'].value_counts()[0]
    initial_bias = np.log([malignant_cases/benign_cases])

    return train, val, initial_bias
