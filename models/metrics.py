import keras.backend as K
from keras.losses import categorical_crossentropy
from keras import metrics
import numpy as np


def c_binary_accuracy(y, p):
    return metrics.binary_accuracy(y[...,0],K.round(p[...,1]))
