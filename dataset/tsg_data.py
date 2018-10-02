from keras.preprocessing import image
import numpy as np

class TSGSaltDataset:
    def __init__(self, train_data_path, val_data_path, batch_size=16, seed=1):
        self.name = 'TSGSaltDataset'
        self.train_path = train_data_path
        self.val_path = val_data_path
        self.batch_size = batch_size
        self.seed = seed
        
        self.augment_args = dict(
            horizontal_flip=True,
            vertical_flip=True
        )
        
    def get_data_generator(self, x_target_size, mask_target_size, source):
        if source == 'train':
            source_path = self.train_path
        elif source == 'test':
            source_path = self.val_path
        else:
            raise ValueError('No such type')
        x_gen = image.ImageDataGenerator(**self.augment_args, 
                                             samplewise_center=True,
                                             samplewise_std_normalization=True)
        x_iter = x_gen.flow_from_directory('{}/images'.format(source_path),
                                           batch_size=self.batch_size,
                                           target_size=x_target_size, color_mode='grayscale', class_mode=None, seed=self.seed)
        
        masks_gen = image.ImageDataGenerator(**self.augment_args, preprocessing_function=self.normalize)
        masks_iter = masks_gen.flow_from_directory('{}/masks'.format(source_path),
                                           batch_size=self.batch_size,
                                           target_size=mask_target_size, color_mode='grayscale', class_mode=None, seed=self.seed)
        
        class_gen = image.ImageDataGenerator(**self.augment_args, preprocessing_function=self.normalize)
        class_iter = masks_gen.flow_from_directory('{}/masks'.format(source_path),
                                           batch_size=self.batch_size,
                                           target_size=mask_target_size, color_mode='grayscale', class_mode=None, seed=self.seed)
        if source == 'train':
            self.train_step_size=len(x_iter)
        else:
            self.val_step_size=len(x_iter)
        
        return zip(x_iter, masks_iter)
        
        
    def normalize(self, image):
        mask = np.where(image > 127,1,0)
        return mask
    
    def has_salt(self, image):
        return np.sum(image) > 0
    
    def normalize_range(self, image):
        return image/256
    
    def has_salt_norm(self, image):
        mask = np.where(image > 127,1,0)
        has_salt = 1 if np.sum(mask) > 0 else 0
        return has_salt