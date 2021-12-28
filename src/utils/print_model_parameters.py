import numpy as np
import tensorflow.keras.backend as K


def print_model_parameters(model):
    trainable_count = np.sum([K.count_params(w)
                              for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w)
                                  for w in model.non_trainable_weights])
    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
