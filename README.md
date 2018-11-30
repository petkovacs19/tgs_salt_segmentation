# TGS Salt Segmentation competition

This is a project containing source code to the [TGS](https://www.kaggle.com/c/tgs-salt-identification-challenge) image segmentation competition.


### Installing

To run training on multiple GPUs, first follow the guides to install [Horovod](https://github.com/uber/horovod). Horovod allows you to run training in a distributed ring-allreduce fashion. [Here](https://www.oreilly.com/ideas/distributed-tensorflow) is a short summary, about ring-allreduce and other distributed architectures.

### Run in distributed mode

Horovod uses [MPI](https://www.open-mpi.org/) - Message Passing Interface under the hood, to communicate between GPUs. A basic intro can be found [here](https://github.com/uber/horovod/blob/master/docs/concepts.md).

```
mpirun -np 3 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib /home/pkovacs/anaconda3/envs/exp/bin/python tsg.py --hvd=True --train_path=/home/pkovacs/tsg/data/train --val_path=/home/pkovacs/tsg/data/val --model=resnet34 --batch_size=16 --epochs=90
```

### Run in single GPU mode

--hvd arg is false by default, so to run in single GPU way, just remove --hvd=True 

```
python tsg.py  --train_path=/home/pkovacs/tsg/data/train --val_path=/home/pkovacs/tsg/data/val --model=resnet34 --batch_size=16 --epochs=90
```



### Generate predictions

Explain what these tests test and why

```
python tsg_predict.py --file_name='path_to_model' --model_name='model_to_load-see_models/models_factory.py'
--target_size=shape_of_input_images
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


