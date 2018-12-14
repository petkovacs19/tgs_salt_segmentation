# TGS Salt Segmentation competition

This is a project containing source code to the [TGS](https://www.kaggle.com/c/tgs-salt-identification-challenge) image segmentation competition.

(!I misspelled TGS, and only realized two months after. You will see TSG at many places in the code and in file names.)

### Installing

To run training on multiple GPUs, first follow the guides to install [Horovod](https://github.com/uber/horovod). Horovod allows you to run training in a distributed ring-allreduce fashion. [Here](https://www.oreilly.com/ideas/distributed-tensorflow) is a short summary, about ring-allreduce and other distributed architectures.


### Stratified k-fold cross validation

The script below will split the dataset into k groups, and create symlinks in the 'folds' folder.
Stratification by coverage is used to ensure each group has a good representation of the whole dataset.

To experiment with other qualities to stratify by, you can extend the TGSDatasetPreprocessor class. Stratified k-fold cross validation helps to better estimate the performance of the trained model on the unseen test set.

```
python tgs_preprocess.py
```

Further experiments: Coverage class may not be the best quality to stratify by and create a good representation of the overall dataset. It may ignore other important features.

### Run in distributed mode
(!Under construction)

Horovod uses [MPI](https://www.open-mpi.org/) - Message Passing Interface under the hood, to communicate between GPUs. A basic intro can be found [here](https://github.com/uber/horovod/blob/master/docs/concepts.md).

```
mpirun -np 3 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib /home/pkovacs/anaconda3/envs/exp/bin/python tsg_train_ring.py --hvd=True --data_path=folds/fold_0 --model=resnet34 --batch_size=16 --epochs=90
```

or alternatively run the bash script below:
(set the path of your python env in the script file)

```
./train_multi_gpu_ring.sh {number_of_gpus}
```

### Run in single-gpu mode

```
python tgs_train.py
```


### Generate predictions

Submission file will be saved in the submission folder in a 'submission_weightFileName_dateTime.csv' file

```
python tgs_predict.py --file_name=path_to_the_model
```

To generate a weighted average prediction of folds, run:

```
python tgs_predict.py --use_folds=True
```

This picks up all the weights from the 'weights/model_name/' folder and generate an avaraged prediction.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


