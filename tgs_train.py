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
        input_shape = mask_shape = (args.target_size, args.target_size)
        train_data_generator = dataset.get_train_data_generator(input_size=input_shape, mask_size=mask_shape, seed=args.seed)
        val_data_generator = dataset.get_val_data_generator(input_size=input_shape, mask_size=mask_shape, seed=args.seed)

        #Find best saved model
        best_model_file = 'weights/{}/fold_{}_{epoch}_best.h5'.format(args.model, fold_index, epoch='{epoch}')
        resume_from_epoch = 0
        for try_epoch in range(args.epochs, 0, -1):
            if os.path.exists(best_model_file.format(epoch=try_epoch)):
                resume_from_epoch = try_epoch
                break
    
        #Create model
        model = make_model(args.model, (args.target_size, args.target_size, 3), 2)

        if resume_from_epoch > 0:
            print("Resuming from epoch {}".format(resume_from_epoch))
            model.load_weights(best_model_file.format(epoch=resume_from_epoch))
        #Optimizer
        opt = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True)

        #Compile model
        model.compile(loss=losses.c_binary_crossentropy,
                      optimizer=opt,
                      metrics=[metrics.c_binary_accuracy, metrics.c_iou])
        
        #Keras callbacks
        callbacks = [
            keras.callbacks.TensorBoard(args.log_dir),
            keras.callbacks.ModelCheckpoint(best_model_file, save_best_only=True, save_weights_only=True)
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

        #Lovasz Loss
        model_with_lovasz = make_model(args.model, (args.target_size, args.target_size, 3), 2)
        
        #Load weights
        resume_from_epoch = 0
        for try_epoch in range(args.epochs, 0, -1):
            if os.path.exists(best_model_file.format(epoch=try_epoch)):
                resume_from_epoch = try_epoch
                break
        model_with_lovasz.load_weights(best_model_file.format(epoch=resume_from_epoch), by_name=True)

        #Optimizer
        opt = SGD(lr=args.learning_rate, momentum=0.9, nesterov=True)
        
        model_with_lovasz.compile(loss=losses.c_lovasz_loss,
                                  optimizer=opt,
                                  metrics=[metrics.c_binary_accuracy, metrics.c_iou])

        print("Fine tuning with lovasz loss")
        model_with_lovasz.fit_generator(train_data_generator,
                            steps_per_epoch=train_step_size,
                            callbacks=callbacks,
                            epochs=args.epochs*2,
                            verbose=args.v,
                            workers=4,
                            initial_epoch=resume_from_epoch,
                            validation_data=val_data_generator,
                            validation_steps=val_step_size)   
        

        # Evaluate the model on the validation data set.
        score = model.evaluate_generator(val_data_generator, val_step_size)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


if __name__== "__main__":
    parser = argparse.ArgumentParser(description = 'TGSSaltModel')
    parser.add_argument('--model', type=str, help='Name of backbone architecture', default="resnet34")
    parser.add_argument('--log_dir', type=str, help='Directory to save logs', default='./logs')
    parser.add_argument('--verbose', type=int, help='Verbose mode', default=1)
    parser.add_argument('--epochs', type=int, help='Number of epochs to run the training for', default=10)
    parser.add_argument('--batch_size', type=int, help='Data batch size', default=32)
    parser.add_argument('--seed', type=int, help='Seed value for data generator', default=1)
    parser.add_argument('--data_path', type=str, help='Path to data folds', default='folds')
    parser.add_argument('--resume_from_epoch', type=int, help='Epoch to resume from', default=0)
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.01)
    parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5', help='checkpoint file format')
    parser.add_argument('--warmup_epochs', type=float, default=5, help='number of warmup epochs')
    parser.add_argument('--target_size', type=int, default=224, help='Target size for images to scale to')
    parser.add_argument('--v', type=int, default=1, help='Verbose mode for keras')
    args = parser.parse_args()
    if not os.path.exists('weights'):
        os.makedirs('weights/{}'.format(args.model))
    main(args)