import argparse
import horovod.keras as hvd
import tensorflow as tf
import keras
import os
from keras import backend as K
from models.model_factory import make_model
from dataset.tgs_data import TGSDataset
from keras.metrics import binary_accuracy
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from models import losses
from models import metrics


class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


def main(args):    
    if args.hvd:
        #initialize Horovod.
        hvd.init()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())
        K.set_session(tf.Session(config=config))
    
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    
    #Create model
    model = make_model(args.model, (args.target_size, args.target_size, 3), 2)
    
    #Resume from epoch
    if args.resume_from_epoch > 0:
        raise ValueError('Not implemented yet')
    else:
        size = hvd.size() if args.hvd else 1
        opt = SGD(lr=args.learning_rate * size, momentum=0.9, nesterov=True)
        #Wrap optimizer in distributed horovod wrapper
        if args.hvd:
            opt = hvd.DistributedOptimizer(opt)

        model.compile(loss=losses.c_binary_crossentropy,
                       optimizer=opt,
                       metrics=[metrics.c_binary_accuracy]) #hard_dice_coef_ch1, hard_dice_coef])
        
    #verbose mode
    if args.hvd and hvd.rank()==0:
        verbose = 1
    elif args.hvd == False:
        verbose = 1
    else:
        verbose = 0
   
    #Creating dataset
    dataset = TGSDataset(train_data_path=args.train_path, val_data_path=args.val_path, batch_size=args.batch_size, seed=args.seed)
    input_shape = (args.target_size, args.target_size)
    mask_shape = (args.target_size, args.target_size)
    train_data_generator = dataset.get_train_data_generator(input_size=input_shape, mask_size=mask_shape)
    val_data_generator = dataset.get_val_data_generator(input_size=input_shape, mask_size=mask_shape)
    
    #h5 model
    best_model_file = '{}_best.h5'.format(args.model)
    best_model = ModelCheckpointMGPU(model, filepath=best_model_file, monitor='val_loss',
                                     verbose=1,
                                     mode='min',
                                     period=1,
                                     save_best_only=True,
                                     save_weights_only=True)
    if args.hvd:
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
    else:
        callbacks = [
            keras.callbacks.TensorBoard(args.log_dir),
            best_model
        ]

    train_step_size = dataset.train_step_size // hvd.size() if args.hvd else dataset.train_step_size
    val_step_size = dataset.val_step_size // hvd.size() if args.hvd else dataset.val_step_size
    model.fit_generator(train_data_generator,
                        steps_per_epoch=train_step_size,
                        callbacks=callbacks,
                        epochs=args.epochs,
                        verbose=verbose,
                        workers=4,
                        initial_epoch=resume_from_epoch,
                        validation_data=val_data_generator,
                        validation_steps=val_step_size)

    # Evaluate the model on the validation data set.
    if hvd:
        score = hvd.allreduce(model.evaluate_generator(val_data_generator, len(test_iter), workers=4))
    else:
        model.evaluate_generator(val_data_generator, len(test_iter), workers=4)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__== "__main__":
    parser = argparse.ArgumentParser(description = 'TGSSaltModel')
    parser.add_argument('--hvd', type=bool, help='If true it will run in Horovod distributed mode', default=False) 
    parser.add_argument('--model', type=str, help='Name of backbone architecture', default="resnet34")
    parser.add_argument('--log_dir', type=str, help='Directory to save logs', default='./logs')
    parser.add_argument('--verbose', type=int, help='Verbose mode', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs to run the training for', default=90)
    parser.add_argument('--batch_size', type=int, help='Data batch size', default=32)
    parser.add_argument('--seed', type=int, help='Seed value for data generator', default=1)
    parser.add_argument('--train_path', type=str, help='Path to the training data', default='/home/pkovacs/tsg/data/train')
    parser.add_argument('--val_path', type=str, help='Path to the val data', default='/home/pkovacs/tsg/data/val')
    parser.add_argument('--resume_from_epoch', type=int, help='Epoch to resume from', default=0)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.0005)
    parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5', help='checkpoint file format')
    parser.add_argument('--warmup_epochs', type=float, default=5, help='number of warmup epochs')
    parser.add_argument('--target_size', type=int, default=224, help='Target size for images to scale to')
    args = parser.parse_args()
    main(args)