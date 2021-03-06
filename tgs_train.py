import argparse
import horovod.keras as hvd
import tensorflow as tf
import keras
from keras import Model
import os
from keras import backend as K
from models.model_factory import make_model
from dataset.tgs_data import TGSDataset
from keras.metrics import binary_accuracy
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from models import losses
from models import metrics
from keras.models import load_model
import gc
    
def main(args):
    
    num_of_folds = len(os.listdir(args.data_path)) 
    print("Found {} number of folds".format(num_of_folds))
    for fold_index in range(num_of_folds):
        print("================================")
        print("Starting fold {}".format(fold_index))
        
        
        #Create dataset
        dataset = TGSDataset(data_path="{}/fold_{}".format(args.data_path, fold_index), batch_size=args.batch_size)
        input_shape = (args.target_size, args.target_size)
        mask_shape = (101, 101)
        train_data_generator = dataset.get_train_data_generator(input_size=input_shape, mask_size=mask_shape, seed=args.seed)
        val_data_generator = dataset.get_val_data_generator(input_size=input_shape, mask_size=mask_shape, seed=args.seed)

        #Find best saved model
        best_model_file = 'weights/{}/fold_{}_{epoch}_best.h5'.format(args.model, fold_index, epoch='{epoch}')
        resume_from_epoch = 0
        for try_epoch in range(args.epochs, 0, -1):
            if os.path.exists(best_model_file.format(epoch=try_epoch)):
                resume_from_epoch = try_epoch
                break
    
        
        
        if resume_from_epoch > 0:
            print("Resuming from epoch {}".format(resume_from_epoch))
            model = load_model(best_model_file.format(epoch=resume_from_epoch),
                              custom_objects={'c_iou':metrics.c_iou})
        else:
            model = make_model(args.model, (args.target_size, args.target_size, 3), 1)

            
        #Optimizer
        opt = adam(lr=args.learning_rate)

        #Compile model
        model.compile(loss=binary_crossentropy,
                      optimizer=opt,
                      metrics=[binary_accuracy, metrics.c_iou])
        
        #Keras callbacks
        callbacks = [
            keras.callbacks.TensorBoard(args.log_dir),
            keras.callbacks.ModelCheckpoint(best_model_file, save_best_only=True, save_weights_only=False),
            keras.callbacks.EarlyStopping(monitor='c_iou', patience=20, verbose=0, mode='max')
        ]
        
        train_step_size = dataset.train_step_size
        val_step_size = dataset.val_step_size

        history = model.fit_generator(train_data_generator,
                            steps_per_epoch=train_step_size,
                            callbacks=callbacks,
                            epochs=args.epochs,
                            verbose=args.v,
                            workers=4,
                            initial_epoch=resume_from_epoch,
                            validation_data=val_data_generator,
                            validation_steps=val_step_size)


        
        #Load weights
        resume_from_epoch = 0
        for try_epoch in range(args.epochs, 0, -1):
            if os.path.exists(best_model_file.format(epoch=try_epoch)):
                resume_from_epoch = try_epoch
                break
        
        if resume_from_epoch > 0:
            print("Resuming from epoch {}".format(resume_from_epoch))
            model_with_lovasz = load_model(best_model_file.format(epoch=resume_from_epoch), custom_objects={"c_iou":metrics.c_iou})
        else:
            model_with_lovasz = make_model(args.model, (args.target_size, args.target_size, 3), 1)
            #Lovasz Loss
            
        #Optimizer
        #Keras callbacks
        callbacks = [
            keras.callbacks.TensorBoard(args.log_dir),
            keras.callbacks.ModelCheckpoint(best_model_file, save_best_only=True, save_weights_only=False),
            keras.callbacks.EarlyStopping(monitor='c_iou_zero', mode='max', patience=20, verbose=0)
        ]
        
        train_data_generator = dataset.get_train_data_generator(input_size=input_shape, mask_size=mask_shape, seed=args.seed)
        val_data_generator = dataset.get_val_data_generator(input_size=input_shape, mask_size=mask_shape, seed=args.seed)

        model_with_lovasz = Model(model_with_lovasz.layers[0].input, model_with_lovasz.layers[-1].input)
        opt = adam(lr=args.learning_rate)
        model_with_lovasz.compile(loss=losses.c_lovasz_loss,
                                  optimizer=opt,
                                  metrics=[binary_accuracy, metrics.c_iou_zero])
        print("Fine tuning with lovasz loss")
        model_with_lovasz.fit_generator(train_data_generator,
                            steps_per_epoch=train_step_size,
                            callbacks=callbacks,
                            epochs=args.epochs,
                            verbose=args.v,
                            workers=4,
                            initial_epoch=resume_from_epoch,
                            validation_data=val_data_generator,
                            validation_steps=val_step_size)   
        

        # Evaluate the model on the validation data set.
        score = model_with_lovasz.evaluate_generator(val_data_generator, val_step_size)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


if __name__== "__main__":
    parser = argparse.ArgumentParser(description = 'TGSSaltModel')
    parser.add_argument('--model', type=str, help='Name of backbone architecture', default="custom_resnet")
    parser.add_argument('--log_dir', type=str, help='Directory to save logs', default='./logs')
    parser.add_argument('--verbose', type=int, help='Verbose mode', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs to run the training for', default=100)
    parser.add_argument('--batch_size', type=int, help='Data batch size', default=16)
    parser.add_argument('--seed', type=int, help='Seed value for data generator', default=1)
    parser.add_argument('--data_path', type=str, help='Path to data folds', default='folds')
    parser.add_argument('--resume_from_epoch', type=int, help='Epoch to resume from', default=0)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.005)
    parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5', help='checkpoint file format')
    parser.add_argument('--warmup_epochs', type=float, default=5, help='number of warmup epochs')
    parser.add_argument('--target_size', type=int, default=101, help='Target size for images to scale to')
    parser.add_argument('--v', type=int, default=1, help='Verbose mode for keras')
    args = parser.parse_args()
    if not os.path.exists('weights'):
        os.makedirs('weights/{}'.format(args.model))
    main(args)