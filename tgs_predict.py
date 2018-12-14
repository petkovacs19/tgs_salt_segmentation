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
from keras.models import load_model

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
    
def create_submission_file(predictions, filenames):
    """
    predictions - list of preditictions on test set
    filenames - list of test files
    name - name of submission to save to
    """
    print("Generating rle encoding")
    lines = []
    for index,prediction in enumerate(tqdm(predictions)):
        prediction = np.array(prediction[...,1])
        rle = rle_encoding(np.round(prediction))
        lines.append("{},{}".format(filenames[index][5:][:-4], rle))
    submission = "id,rle_mask\n" + "\n".join(lines)
    file_path = 'submissions/{}/submission_{}_{}.csv'.format(args.model_name,args.weight_path[0].split("/")[-1],datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    with open(file_path, "w") as f:
        f.write(submission)
    return file_path

def generate_predictions(file_name, model_name, target_size, test_iter):
    """
    Using model - model_name and saved weights in file_name, generates predictions for files in test_path
    """
    weight = os.path.join('{}.h5'.format(file_name))
    model = make_model(model_name, (target_size, target_size, 3), 2)
    model.load_weights(weight, by_name=True)
    #model = load_model(weight)
    predictions = model.predict_generator(test_iter, steps = len(test_iter.filenames), verbose=1)
    return predictions

def generate_submission_file(weight_path, model_name, target_size, test_path, use_folds):
    """
    Generates rle encoded submission files of model - model_name
    use_folds - if True - a weighted average prediction of folds is used from weight files in folder - 'weights/model_name'
              - if False - a single weight file - file_name is used 
    """
    test_gen = image.ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
    test_iter = test_gen.flow_from_directory(test_path, batch_size=1, target_size=(target_size, target_size),
                                             class_mode=None,seed=1, shuffle = False)
    predictions = np.zeros((18000, 224, 224, 2))
    if use_folds:
        for _,_,files in os.walk('weights/{}'.format(model_name)):
            weight_files = files
    else:
        weight_files = weight_path
        
    for weight_file in weight_files:
        print("Generating prediction for model {} and weight {}".format(model_name, weight_file))
        predictions += generate_predictions(weight_file, model_name, target_size, test_iter) / len(weight_files)     
    
    return create_submission_file(predictions, test_iter.filenames)
   
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'This is to generate submission file for the predictions')
    parser.add_argument('--weight_path', nargs='+', help='path to the best model', default="resnet34_best")
    parser.add_argument('--model_name', type=str, help='model to load', default="resnet34")
    parser.add_argument('--use_folds', type=bool, help='use average prediction of cross-validation folds', default=False)
    parser.add_argument('--target_size', type=int, help='target size', default=224)
    parser.add_argument('--test_path', type=str, help='Path to the test data', default='/home/pkovacs/tsg/data/test/images')
    args = parser.parse_args()
    print("==================================================")
    print("Generating predictions for test set using")
    print("Model: {}".format(args.model_name))
    print("Weights: {}".format(args.weight_path))
    print("Using data: {}".format(args.test_path))
    print("==================================================")
    if not os.path.exists('submissions'):
        os.makedirs('submissions/{}'.format(args.model_name))
    submission_file_path = generate_submission_file(args.weight_path, args.model_name, args.target_size, args.test_path, args.use_folds)
    print("Submission saved at {}".format(submission_file_path))