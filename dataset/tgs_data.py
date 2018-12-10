from keras.preprocessing import image
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import os


class TGSDataset:
    def __init__(self, data_path, batch_size=16, seed=1, kfold=1):
        self.name = 'TGSDataset'
        self.data_path = data_path
        self.batch_size = batch_size
        self.seed = seed
        
    def get_data_generator(self, dataframe, input_size, mask_size, validation=False):
        """
        input_size - tuple(int,int) width,height of input to the model
        mask_size - tuple(int,int) expected width,height of output mask from the model
        """
        x_gen = image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
        x_iter = x_gen.flow_from_dataframe(dataframe, '{}/images/salt'.format(self.data_path), x_col='file', y_col='class',
                                           batch_size=self.batch_size,
                                           target_size=input_size,
                                           class_mode=None,
                                           seed=self.seed)
        
        y_masks_gen = image.ImageDataGenerator(preprocessing_function=self.normalize)
        y_masks_iter = y_masks_gen.flow_from_dataframe(dataframe, '{}/masks/salt'.format(self.data_path), x_col='file', y_col='class',
                                                       batch_size=self.batch_size,
                                                       target_size=mask_size,
                                                       color_mode='grayscale',
                                                       class_mode=None,
                                                       seed=self.seed)
        return self.generate_data_generator(x_iter, y_masks_iter)
           
    def normalize(self, image):
        mask = np.where(image > 127,1,0)
        return mask
    
    def has_salt_norm(self, image):
        mask = np.where(image > 127,1,0)
        has_salt = 1 if np.sum(mask) > 0 else 0
        return has_salt  
    
    def generate_data_generator(self, x_generator, y_generator):
        while True:
                x = x_generator.next()
                mask = y_generator.next()
                yield x, mask
    
    
class TGSDatasetPreprocessor:
    def __init__(self, data_path, seed=1):
        """
        This is for generating k-folds from the dataset
        """
        self.data_path = data_path
        self.seed = 1
        
        
    def get_data_with_features(self):
        """
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
        
        self.filenames = list(map(lambda x: x.split("/")[1], x_iter.filenames))
        file_classes = [np.squeeze(y_masks_iter.next())[0][0] for l in self.filenames]
        return pd.DataFrame(data= {'file': self.filenames, 'class': file_classes})
       
    def coverage_class(self, image):
        """
        Coverage class used for stratified k-fold cross validation
        """
        mask = np.where(image > 127, 1, 0)
        coverage = int(np.sum(mask) / (image.shape[0]*image.shape[1]) * 10)
        return coverage