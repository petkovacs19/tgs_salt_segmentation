from keras.preprocessing import image
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os

class TGSDataset:
    def __init__(self, data_path, batch_size=16, kfold=1):
        self.name = 'TGSDataset'
        self.data_path = data_path
        self.batch_size = batch_size
        
    def get_train_data_generator(self, input_size, mask_size, seed=1):
        """
        input_size - tuple(int,int) width,height of input to the model
        mask_size - tuple(int,int) expected width,height of output mask from the model
        """
        return self.__get_data_generator(input_size, mask_size, False, seed)
    
        
    def get_val_data_generator(self, input_size, mask_size, seed=1):
        return self.__get_data_generator(input_size, mask_size, True, seed)
        
    def __get_data_generator(self, input_size, mask_size, validation=False, seed=1):
        tpe = 'val' if validation else 'train'
        x_gen = image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
        x_iter = x_gen.flow_from_directory('{}/{}/images'.format(self.data_path, tpe),
                                           batch_size=self.batch_size,
                                           target_size=input_size,
                                           class_mode=None,
                                           seed=seed)
        
        y_masks_gen = image.ImageDataGenerator(preprocessing_function=self.normalize)
        y_masks_iter = y_masks_gen.flow_from_directory('{}/{}/masks'.format(self.data_path, tpe),
                                                       batch_size=self.batch_size,
                                                       target_size=mask_size,
                                                       color_mode='grayscale',
                                                       class_mode=None,
                                                       seed=seed)
        if validation:
            self.val_step_size=len(x_iter)
        else:
            self.train_step_size=len(x_iter)
        return zip(x_iter, y_masks_iter)
           
    def normalize(self, image):
        mask = np.where(image > 127,1,0)
        return mask
    
    def has_salt_norm(self, image):
        mask = np.where(image > 127,1,0)
        has_salt = 1 if np.sum(mask) > 0 else 0
        return has_salt    
    
    
class TGSDatasetPreprocessor:
    def __init__(self, data_path, seed=1):
        """
        This is for generating k-folds from the dataset
        """
        self.data_path = data_path
        self.seed = 1
        
        
    def gen_k_folds(self, num_of_folds):
        """
        Generates folds and saves them to fold folder
        num_of_folds - int - number of folds to generate
        """
        
        x_gen = image.ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
        x_iter = x_gen.flow_from_directory('{}/images/'.format(self.data_path),
                                         batch_size=1,
                                         target_size=(224, 224), class_mode=None,seed=self.seed, shuffle = False)
        
        y_masks_gen = image.ImageDataGenerator(preprocessing_function=self.coverage_class)
        y_masks_iter = y_masks_gen.flow_from_directory('{}/masks'.format(self.data_path),
                                                       batch_size=1,
                                                       target_size=(224, 224),
                                                       color_mode='grayscale',
                                                       class_mode=None,
                                                       seed=self.seed)
        
        self.filenames = x_iter.filenames
        file_classes = [np.squeeze(y_masks_iter.next())[0][0] for l in self.filenames]
        return StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=self.seed).split(self.filenames, file_classes)
    
    def coverage_class(self, image):
        """
        Coverage class used for stratified k-fold cross validation
        """
        mask = np.where(image > 127, 1, 0)
        coverage = int(np.sum(mask) / (image.shape[0]*image.shape[1]) * 10)
        return coverage




        

    
    
    
