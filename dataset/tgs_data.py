from keras.preprocessing import image
import numpy as np

class TGSDataset:
    def __init__(self, train_data_path, val_data_path, batch_size=16, seed=1):
        self.name = 'TGSDataset'
        self.train_path = train_data_path
        self.val_path = val_data_path
        self.batch_size = batch_size
        self.seed = seed
        
    def get_train_data_generator(self, input_size, mask_size):
        """
        input_size - tuple(int,int) width,height of input to the model
        mask_size - tuple(int,int) expected width,height of output mask from the model
        """
        return self.__get_data_generator(input_size, mask_size)
    
        
    def get_val_data_generator(self, input_size, mask_size):
        return self.__get_data_generator(input_size, mask_size, True)
        
    def __get_data_generator(self, input_size, mask_size, validation=False):
        path = self.val_path if validation else self.train_path
        x_gen = image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
        x_iter = x_gen.flow_from_directory('{}/images'.format(path),
                                           batch_size=self.batch_size,
                                           target_size=input_size,
                                           class_mode=None,
                                           seed=self.seed)
        
        y_masks_gen = image.ImageDataGenerator(preprocessing_function=self.normalize)
        y_masks_iter = y_masks_gen.flow_from_directory('{}/masks'.format(path),
                                                       batch_size=self.batch_size,
                                                       target_size=mask_size,
                                                       color_mode='grayscale',
                                                       class_mode=None,
                                                       seed=self.seed)
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