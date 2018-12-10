import os
import argparse
from dataset.tgs_data import TGSDataset
from dataset.tgs_data import TGSDatasetPreprocessor
import shutil
import numpy as np


def create_sym_links(files, fold_index, part='train'):
    """
    Creates symlinks for files in folder 'folds'
    files - list files to create symlinks for
    fold_index - int - symlinks will be created in 'folds/fold_{}/'
    part - str - subdir to place symlinks to/ train/val
    """
    print("Creating symlinks for fold {} and files for {} set".format(fold_index, part))
    for file in files:
        os.symlink("{}/images/{}".format(args.data_path,file), 'folds/fold_{}/{}/images/{}'.format(fold_index, part, file))
        os.symlink("{}/masks/{}".format(args.data_path,file), 'folds/fold_{}/{}/masks/{}'.format(fold_index, part, file))

def generate_folds(folds):
    """
    folds - int - number of folds to generate
    Generate k-fold from dataset and save them to fold_{} folder
    """
    pre = TGSDatasetPreprocessor(args.data_path)
    folds = pre.gen_k_folds(args.fold)
    files = pre.filenames
    for index,fold in enumerate(folds):
        for elem in ['train', 'val']:
            os.makedirs('folds/fold_{}/{}/images/salt'.format(index, elem))
            os.makedirs('folds/fold_{}/{}/masks/salt'.format(index, elem))
        create_sym_links(np.array(files)[fold[1]], index, 'val')
        create_sym_links(np.array(files)[fold[0]], index, 'train')
             
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'This is to split training data using k-fold validation. This script will generate folders for each fold and create symlinks to the given subset of dataset')
    parser.add_argument('--fold', type=int, help='Number of folds', default=5)
    parser.add_argument('--data_path', type=str, help='Path to data', default='/home/pkovacs/tsg/data/all')
    args = parser.parse_args()
    print("==================================================")
    print("Generating {} folds of the data".format(args.fold))
    print("==================================================")
    if os.path.exists('folds'):
        shutil.rmtree('folds')
    os.makedirs('folds')
    generate_folds(args.fold)