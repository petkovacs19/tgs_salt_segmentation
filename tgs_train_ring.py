import argparse
import horovod.keras as hvd
import tensorflow as tf
import keras
import os
import numpy as np
from keras import backend as K
from models.model_factory import make_model
from dataset.tgs_data import TGSDataset
from keras.metrics import binary_accuracy
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from models import losses
from models import metrics
from keras.models import load_model
import gc

class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


def main(args):    
    #initialize Horovod.
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))
    
    fold = args.data_path.split("fold_")[1]
    if hvd.rank()==0:
        print("================================")
        if args.use_lovasz:
            print("Fine tuning with ")
        print("Fold {}".format(fold))
        
    #Find best saved model
    best_model_file = 'weights/{}/fold_{}_{epoch}_best.h5'.format(args.model, fold, epoch='{epoch}')
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(best_model_file.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break
    if hvd.rank()==0:
        print("Last model saved: {}".format(best_model_file.format(epoch=resume_from_epoch)))
        
    #verbose mode for one node
    if hvd.rank()==0:
        verbose = 1
    else:
        verbose = 0
   
    #Create dataset
    
    dataset = TGSDataset(data_path=args.data_path, batch_size=args.batch_size)
    input_shape = mask_shape = (args.target_size, args.target_size)
    train_data_generator = dataset.get_train_data_generator(input_size=input_shape, mask_size=mask_shape, seed=np.random.rand())
    val_data_generator = dataset.get_val_data_generator(input_size=input_shape, mask_size=mask_shape, seed=np.random.rand())
    train_step_size = dataset.train_step_size // hvd.size()
    val_step_size = dataset.val_step_size // hvd.size()
    #Create model
    model = make_model(args.model, (args.target_size, args.target_size, 3), 2)

    #load weights
    if resume_from_epoch > 0:
        model.load_weights(best_model_file.format(epoch=resume_from_epoch))
        
    size = hvd.size()
    opt = hvd.DistributedOptimizer(SGD(lr=args.learning_rate * size, momentum=0.9, nesterov=True))

    #Loss
    loss = losses.c_lovasz_loss if args.use_lovasz else losses.c_binary_crossentropy
    
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=[metrics.c_binary_accuracy, metrics.c_iou])

    #h5 model
    best_model = ModelCheckpointMGPU(model, filepath=best_model_file, monitor='val_loss',
                                     verbose=1,
                                     mode='min',
                                     period=1,
                                     save_best_only=True,
                                     save_weights_only=True)
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard, or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=args.warmup_epochs, verbose=True)
    ]

    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.TensorBoard(args.log_dir))
        callbacks.append(best_model)
    
    #Fit model
    history = model.fit_generator(train_data_generator,
                        steps_per_epoch=train_step_size,
                        callbacks=callbacks,
                        epochs=args.epochs,
                        verbose=verbose,
                        workers=4,
                        initial_epoch=resume_from_epoch,
                        validation_data=val_data_generator,
                        validation_steps=val_step_size)
  

    score = hvd.allreduce(model.evaluate_generator(val_data_generator, val_step_size, workers=4))
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__== "__main__":
    parser = argparse.ArgumentParser(description = 'TGSSaltModel')
    parser.add_argument('--model', type=str, help='Name of backbone architecture', default="resnet34")
    parser.add_argument('--log_dir', type=str, help='Directory to save logs', default='./logs')
    parser.add_argument('--verbose', type=int, help='Verbose mode', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs to run the training for', default=150)
    parser.add_argument('--batch_size', type=int, help='Data batch size', default=32)
    parser.add_argument('--data_path', type=str, help='Path to data folds', default='folds')
    parser.add_argument('--resume_from_epoch', type=int, help='Epoch to resume from', default=0)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.0005)
    parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5', help='checkpoint file format')
    parser.add_argument('--warmup_epochs', type=float, default=5, help='number of warmup epochs')
    parser.add_argument('--target_size', type=int, default=224, help='Target size for images to scale to')
    parser.add_argument('--use_lovasz', type=bool, default=False, help="Use lovasz losz")
    args = parser.parse_args()
    if not os.path.exists('weights'):
        os.makedirs('weights/{}'.format(args.model))
    main(args)