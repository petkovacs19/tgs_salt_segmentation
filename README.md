# TSG Salt Segmentation competition

This is a project containing source code to the TSG image segmentation competition.

More to come.


### Run in distributed mode

```
mpirun -np 3 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib /home/pkovacs/anaconda3/envs/exp/bin/python tsg.py --hvd=True --train_path=/home/pkovacs/tsg/data/train --val_path=/home/pkovacs/tsg/data/val --model=resnet34 --batch_size=16 --epochs=90
```

