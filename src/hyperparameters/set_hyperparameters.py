import os
import json


def set_hyperparameters(hyperparameter_dict):
    os.environ["HYPERPARAMETERS"] = json.dumps(hyperparameter_dict)
