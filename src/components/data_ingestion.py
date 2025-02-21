import tensorflow as tf 
import os 
import sys
from dataclasses import dataclass
from pathlib import  Path
import gdown
import zipfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))
from src.logger import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self):
        pass
    @staticmethod
    def load_dataset_from_dir(directory):
       
        
        try:             
             dataset = tf.keras.utils.image_dataset_from_directory(
                    directory, 
                    labels="inferred",               
                    label_mode="int",
                    #class_names=self.class_names,               
                    color_mode="rgb",                  
                    batch_size=None,                     
                    image_size=(256, 256),          
                    shuffle=True,                     
                    seed=123, 
                )
             return dataset
        except Exception as e:
            raise CustomException(e,sys)
    @staticmethod
    def split_dataset(dataset,Train_ratio,Val_ratio,Test_ratio):
       try:
            data_size = len(dataset)
            
            train_dataset = dataset.take(int(Train_ratio*data_size))
            
            test_val_set = dataset.skip(int(Train_ratio*data_size))
            
            valid_dataset = test_val_set.take(int(Val_ratio*data_size))
            
            test_dataset = test_val_set.skip(int(Val_ratio*data_size))
            
            return train_dataset,valid_dataset,test_dataset
       except Exception as e:
           raise CustomException(e,sys)
    @staticmethod
    def normalize_the_dataset(image,label):
        # use augmentation
        image = tf.image.rot90(image)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.adjust_brightness(image,0.3)
        image = tf.image.adjust_contrast(image,0.2)
        image = tf.image.adjust_saturation(image,0.2)
        image = tf.image.adjust_hue(image,0.2)
        return tf.image.resize(image,(224,224))/255,label
    
    def partition_dataset(self,train_dataset,valid_dataset,test_dataset):
        self.train_dataset = (train_dataset
                .shuffle(buffer_size=8,reshuffle_each_iteration=True)
                .map(self.normalize_the_dataset)
                .batch(12)
                .prefetch(tf.data.AUTOTUNE)
                )
        self.valid_dataset = (valid_dataset
                .shuffle(buffer_size=8,reshuffle_each_iteration=True)
                .map(self.normalize_the_dataset)
                .batch(12)
                .prefetch(tf.data.AUTOTUNE)
                )
        self.test_dataset = test_dataset.map(self.normalize_the_dataset).batch(1)
        
        return self.train_dataset,self.valid_dataset,self.test_dataset