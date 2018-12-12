import keras.backend as K
from keras.losses import categorical_crossentropy
from keras import metrics
import numpy as np

def c_binary_accuracy(y_truth, y_pred):
    return metrics.binary_accuracy(y_truth[...,0],K.round(y_pred[...,1]))



def c_iou(y_truth, y_pred):
    t, p = K.flatten(y_truth[...,0])>0.5, K.flatten(y_pred[...,1])>0.5
    intersection = K.cast(K.all(K.stack([t, p], axis=0), axis=0),'int32')
    union = K.cast(K.any(K.stack([t, p], axis=0), axis=0),'int32')
    iou = K.sum(K.cast(intersection > 0,'int32')) / K.sum(K.cast(union > 0,'int32'))
    return K.mean(iou)