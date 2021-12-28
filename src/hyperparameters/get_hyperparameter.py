import os
import json


def get_hyperparameter(parameter):
    return json.loads(os.environ["HYPERPARAMETERS"]).get(parameter)
