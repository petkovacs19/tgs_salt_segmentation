import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.metrics import binary_accuracy
import numpy as np

def c_binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y[...,0],p[...,1]))
