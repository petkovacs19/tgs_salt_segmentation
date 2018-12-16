import keras.backend as K
from keras.losses import categorical_crossentropy
from keras import metrics
import numpy as np
import tensorflow as tf

def c_binary_accuracy(y_truth, y_pred):
    return metrics.binary_accuracy(y_truth[...,0],y_pred[...,0])

def c_iou(y_truth, y_pred):
    t, p = K.flatten(K.round(y_truth[...,0])), K.flatten(K.round(y_pred[...,0]))
    intersection = K.all(K.stack([t, p], axis=0), axis=0)
    union = K.any(K.stack([t, p], axis=0), axis=0)
    iou = K.sum(K.cast(intersection,'int32')) / K.sum(K.cast(union,'int32'))
    return K.mean(iou)


def c_iou_zero(y_truth, y_pred):
    t, p = K.flatten(K.cast(K.round(y_truth[...,0]),'int32')), K.flatten(K.cast(y_pred[...,0] > 0,'int32'))
    intersection = K.all(K.stack([t, p], axis=0), axis=0)
    union = K.any(K.stack([t, p], axis=0), axis=0)
    iou = K.sum(K.cast(intersection,'int32')) / K.sum(K.cast(union,'int32'))
    return K.mean(iou)
