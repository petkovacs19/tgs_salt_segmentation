import os
import argparse
from dataset.tgs_data import TGSDataset
from dataset.tgs_data import TGSDatasetPreprocessor


def create_sym_links(files, fold):
    """
    files - list of files to create symlinks for
    fold - symlinks will be created in 'folds/fold_{}/'
    """
    print("Creating symlinks for fold {}".format(fold))

def generate_folds(folds):
    """
    Number of splits to generate from the dataset
    """
    
    pre = TGSDatasetPreprocessor(args.data_path)
    folds = pre.gen_k_folds(args.fold)
    for index,fold in enumerate(folds):
        #os.mkdir('folds/fold_{}'.format(index))
        create_sym_links(fold[1], index)
        
    print("DONE")
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'This is to split training data using k-fold validation. This script will generate folders for each fold and create symlinks to the given subset of dataset')
    parser.add_argument('--fold', type=int, help='Number of folds', default=5)
    parser.add_argument('--data_path', type=str, help='Path to data', default='/home/pkovacs/tsg/data/val')
    args = parser.parse_args()
    print("==================================================")
    print("Generating {} folds of the data".format(args.fold))
    print("==================================================")
    if not os.path.exists('folds'):
        os.makedirs('folds')
    generate_folds(args.fold)