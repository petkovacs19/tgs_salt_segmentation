import argparse
import horovod.keras as hvd
import tensorflow as tf
import keras
import os
from keras import backend as K
from models.model_factory import make_model
from dataset.tsg_data import TSGSaltDataset
from keras.metrics import binary_accuracy
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from models import losses
from models import metrics
from models.metrics import my_iou_metric_2
from keras.models import Model, load_model, save_model


class ModelCheckpointMGPU(ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        self.original_model = original_model
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs=None):
        self.model = self.original_model
        super().on_epoch_end(epoch, logs)


def main(args):   
    save_model_name = 'best_resnet_34.model'
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
    model = make_model(args.model, (args.target_size, args.target_size, 3), 16)
    
    #Resume from epoch
    if args.resume_from_epoch > 0:
        raise ValueError('Not implemented yet')
    else:
        size = hvd.size() if args.hvd else 1
        opt = Adam(lr=args.learning_rate * size)
        #Wrap optimizer in distributed horovod wrapper
        if args.hvd:
            opt = hvd.DistributedOptimizer(opt)

        model.compile(loss='binary_crossentropy',
                       optimizer=opt,
                       metrics=[metrics.my_iou_metric]) #hard_dice_coef_ch1, hard_dice_coef])
        
    #verbose mode
    if args.hvd and hvd.rank()==0:
        verbose = 1
    elif args.hvd == False:
        verbose = 1
    else:
        verbose = 0
   
    #Creating dataset
    dataset = TSGSaltDataset(train_data_path=args.train_path, val_data_path=args.val_path, batch_size=args.batch_size, seed=args.seed)
    train_data_generator = dataset.get_train_data_generator(x_target_size=(args.target_size, args.target_size),
                                                            mask_target_size=(args.target_size, args.target_size))
    val_data_generator = dataset.get_val_data_generator(x_target_size=(args.target_size, args.target_size),
                                                        mask_target_size=(args.target_size, args.target_size))
    
    #h5 model
    best_model_file = '{}_best.h5'.format(args.model)
    best_model = ModelCheckpoint(save_model_name,monitor='my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    
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
            best_model,
            reduce_lr
        ]

    train_step_size = dataset.train_step_size // hvd.size() if args.hvd else dataset.train_step_size
    val_step_size = dataset.val_step_size // hvd.size() if args.hvd else dataset.val_step_size
    model.fit_generator(train_data_generator,
                        steps_per_epoch=train_step_size,
                        callbacks=callbacks,
                        epochs=150,
                        verbose=verbose,
                        workers=4,
                        initial_epoch=resume_from_epoch,
                        validation_data=val_data_generator,
                        validation_steps=val_step_size)

    
    ##EXTENSION
    model1 = load_model(save_model_name,custom_objects={'my_iou_metric': metrics.my_iou_metric})
    # remove layter activation layer and use losvasz loss
    input_x = model1.layers[0].input

    output_layer = model1.layers[-1].input
    model = Model(input_x, output_layer)
    c = Adam(lr = 0.01)

    # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
    # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
    model.compile(loss=losses.lovasz_loss, optimizer=c, metrics=[metrics.my_iou_metric_2])

    
    early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
    model_checkpoint = ModelCheckpoint(save_model_name,monitor='val_my_iou_metric_2', 
                                       mode = 'max', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    
    model.fit_generator(train_data_generator,
                        steps_per_epoch=train_step_size,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr],
                        epochs=150,
                        verbose=verbose,
                        workers=4,
                        validation_data=val_data_generator,
                        validation_steps=val_step_size)
    ###EXTENSION over
    # Evaluate the model on the validation data set.
#     if hvd:
#         score = hvd.allreduce(model.evaluate_generator(val_data_generator, len(test_iter), workers=4))
#     else:
#         model.evaluate_generator(val_data_generator, len(test_iter), workers=4)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__== "__main__":
    parser = argparse.ArgumentParser(description = 'TSGSaltModel')
    parser.add_argument('--hvd', type=bool, help='If true it will run in Horovod distributed mode', default=False) 
    parser.add_argument('--model', type=str, help='Name of backbone architecture', default="simple_resnet")
    parser.add_argument('--log_dir', type=str, help='Directory to save logs', default='./logs')
    parser.add_argument('--verbose', type=int, help='Verbose mode', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs to run the training for', default=90)
    parser.add_argument('--batch_size', type=int, help='Data batch size', default=32)
    parser.add_argument('--seed', type=int, help='Seed value for data generator', default=1)
    parser.add_argument('--train_path', type=str, help='Path to the training data', default='/home/pkovacs/tsg/data/train')
    parser.add_argument('--val_path', type=str, help='Path to the val data', default='/home/pkovacs/tsg/data/val')
    parser.add_argument('--resume_from_epoch', type=int, help='Epoch to resume from', default=0)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.01)
    parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5', help='checkpoint file format')
    parser.add_argument('--warmup_epochs', type=float, default=5, help='number of warmup epochs')
    parser.add_argument('--target_size', type=int, default=101, help='Target size for images to scale to')
    args = parser.parse_args()
    main(args)