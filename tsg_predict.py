from keras.preprocessing.image import img_to_array, load_img
from keras.applications.imagenet_utils import preprocess_input
from models.model_factory import make_model
from os import path, mkdir, listdir
import numpy as np
import random
import tensorflow as tf
import timeit
import cv2
from tqdm import tqdm
from keras.preprocessing import image
from keras.layers import AveragePooling2D
import datetime
import os
import argparse


# Default RLenc
def rle_encoding(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def generate_submission_file(file_name, model_name):
    weight = os.path.join('{}.h5'.format(file_name))
    model = make_model(model_name, (None, None, 3))
    model.load_weights(weight)
    
    test_gen = image.ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
    test_iter = test_gen.flow_from_directory('test/images',
                                         batch_size=1,
                                         target_size=(256, 256), class_mode=None,seed=1, shuffle = False)
    filenames = test_iter.filenames
    nb_samples = len(filenames)

    predict = model.predict_generator(test_iter,steps = nb_samples)
    lines = []
    for index,prediction in enumerate(tqdm(predict[0])):
        prediction[prediction<0] = 0
        rounded = np.round(prediction[:,:,0])
        rle = rle_encoding(rounded)
        lines.append("{},{}".format(filenames[index][5:][:-4], rle))
    submission = "id,rle_mask\n" + "\n".join(lines)
    file_path = 'submission_{}_{}'.format(file_name,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    with open(file_path, "w") as f:
        f.write(submission)
    return file_path
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'This is to generate submission file for the predictions')
    parser.add_argument('--file_name', help='path to the best model')
    parser.add_argument('--model_name', help='model to load')
    args = parser.parse_args()
    submission_file_path = generate_submission_file(args.file_name,args.model_name)
    print("Submission saved at {}".format(submission_file_path))