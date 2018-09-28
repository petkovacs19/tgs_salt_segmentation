from keras.preprocessing import image
import numpy as np

class TSGSaltDataset:
    def __init__(self, train_data_path, val_data_path, batch_size=16, seed=1):
        self.name = 'TSGSaltDataset'
        self.train_path = train_data_path
        self.val_path = val_data_path
        self.batch_size = batch_size
        self.seed = seed
        
    def get_train_data_generator(self, target_size=(101,101)):
        train_gen = image.ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
        train_iter = train_gen.flow_from_directory('{}/images'.format(self.train_path),
                                           batch_size=self.batch_size,
                                           target_size=target_size, class_mode=None, seed=self.seed)
        
        train_masks_gen = image.ImageDataGenerator(preprocessing_function=self.normalize)
        train_masks_iter = train_masks_gen.flow_from_directory('{}/masks'.format(self.train_path),
                                           batch_size=self.batch_size,
                                           target_size=target_size, color_mode='grayscale', class_mode=None, seed=self.seed)
        self.train_step_size=len(train_iter)
        return zip(train_iter, train_masks_iter)
    
        
    def get_val_data_generator(self, target_size=(101,101)):
        val_gen = image.ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True)
        val_masks_gen = image.ImageDataGenerator(preprocessing_function=self.normalize)

        val_iter = val_gen.flow_from_directory('{}/images'.format(self.val_path),
                                         batch_size=self.batch_size,
                                         target_size=target_size, class_mode=None,seed=self.seed)
        val_masks_iter = val_masks_gen.flow_from_directory('{}/masks'.format(self.val_path),
                                         batch_size=self.batch_size,
                                         target_size=target_size, color_mode='grayscale', class_mode=None, seed=self.seed)
        self.val_step_size=len(val_iter)
        return zip(val_iter, val_masks_iter)
        
        
    def normalize(self, image):
        mask = np.where(image > 127,1,0)
        has_salt = 1 if np.sum(mask) > 0 else 0
        return mask
    

    def has_salt_norm(self, image):
        mask = np.where(image > 127,1,0)
        has_salt = 1 if np.sum(mask) > 0 else 0
        return has_salt