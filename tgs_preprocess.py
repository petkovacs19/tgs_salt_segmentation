import os
import argparse

def generate_folds(folds):
    """
    Number of splits to generate from the dataset
    """
    print("DONE")
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'This is to split training data using k-fold validation. This script will generate folders for each fold and create symlinks to the given subset of dataset')
    parser.add_argument('--fold', type=int, help='Number of folds', default=1)
    parser.add_argument('--train_path', type=str, help='Path to the training data', default='/home/pkovacs/tsg/data/train')
    args = parser.parse_args()
    print("==================================================")
    print("Generating {} folds of the data".format(args.fold))
    print("==================================================")
    if not os.path.exists('folds'):
        os.makedirs('folds')
    generate_folds(args.fold)