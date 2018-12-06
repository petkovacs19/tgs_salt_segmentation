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
from skimage.transform import resize
from skimage import img_as_bool


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

def generate_submission_file(file_name, model_name, target_size, test_path):
    weight = os.path.join('{}.h5'.format(file_name))
    model = make_model(model_name, (target_size, target_size, 3), 2)
    model.load_weights(weight)
    
    test_gen = image.ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
    test_iter = test_gen.flow_from_directory(test_path,
                                         batch_size=1,
                                         target_size=(target_size, target_size), class_mode=None,seed=1, shuffle = False)
    filenames = test_iter.filenames
    nb_samples = len(filenames)

    predictions = model.predict_generator(test_iter,steps = nb_samples, verbose=1)
    lines = []
    print("Generating rle encoding")
    for index,prediction in enumerate(tqdm(predictions)):
        prediction = np.round(prediction[...,1])
        bool_mask = np.where(prediction==1, True, False)
        bool_mask_res = img_as_bool(resize(bool_mask, (101, 101)))
        prediction = np.where(bool_mask_res == True, 1, 0)
        rle = rle_encoding(prediction)
        lines.append("{},{}".format(filenames[index][5:][:-4], rle))
    submission = "id,rle_mask\n" + "\n".join(lines)
    file_path = 'submissions/submission_{}_{}.csv'.format(file_name,datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    with open(file_path, "w") as f:
        f.write(submission)
    return file_path
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'This is to generate submission file for the predictions')
    parser.add_argument('--file_name', type=str, help='path to the best model', default="resnet34_best")
    parser.add_argument('--model_name', type=str, help='model to load', default="resnet34")
    parser.add_argument('--target_size', type=int, help='target size', default=224)
    parser.add_argument('--test_path', type=str, help='Path to the test data', default='/home/pkovacs/tsg/data/test/images')
    args = parser.parse_args()
    print("==================================================")
    print("Generating predictions for test set using")
    print("Model: {}".format(args.model_name))
    print("Weights: {}".format(args.file_name))
    print("Using data: {}".format(args.test_path))
    print("==================================================")
    if not os.path.exists('submissions'):
        os.makedirs('submissions')
    submission_file_path = generate_submission_file(args.file_name, args.model_name, args.target_size, args.test_path)
    print("Submission saved at {}".format(submission_file_path))