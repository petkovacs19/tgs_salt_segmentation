#!/bin/bash
# This will run training in a distributed ring allReduce fashion using horovod See: https://eng.uber.com/horovod/
if [ $# -eq 0 ]
  then
    echo "Specify the number of gpus you want to run on"
    exit 2
fi

echo "Starting distributed training with horovod"

epochs=1
batch_size=32
python_env=/home/pkovacs/anaconda3/envs/exp/bin/python
data_path=folds
num_of_gpus=$1

echo "EPOCH: $epochs"
echo "BATCH SIZE: $batch_size"
echo "GPUs": "$1"

mpirun -np $num_of_gpus -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib $python_env tgs_train.py --hvd=True --data_path=$data_path --model=resnet34 --batch_size=$batch_size --epochs=$epochs
