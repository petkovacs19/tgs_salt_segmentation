import argparse
import os
import skopt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import Optimizer
from skopt.space import Real
from sklearn.externals.joblib import Parallel, delayed
from skopt.benchmarks import branin
from distributed.joblib import DistributedBackend 
from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend
from tqdm import tqdm
import pickle
import time


def main(args):
    """
    Search hyperparameters for resnet model"
    """
    
    ##Specify dimensions
    


if __name__== "__main__":
    parser = argparse.ArgumentParser(description = 'ResnetHyperParameterSearcher')
    args = parser.parse_args()
    main(args)