import os 
import sys
from dataclasses import dataclass
from pathlib import Path
import tensorflow as tf 
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from src.constant import CONFIG_FILE_PATH,PARAMS_FILE_PATH
from src.logger import logging
from src.utils.common import create_directories,read_yaml,dataset_arrangement
from src.exception import CustomException

train_dataset,valid_dataset,test_dataset = dataset_arrangement()
# train_model 
class trainer:
    def __init__(self):
        pass 
    def get_train_model(self):
        model = tf.keras.models.load_model('artifacts/base_model/base_model.h5')
        model.compile(loss  = tf.keras.losses.BinaryCrossentropy(),
                    optimizer = tf.keras.optimizers.Adam(),
                    metrics = ['accuracy'])
        model.summary()
        # track the mdoel 
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=1,
            mode='auto',
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=5,
        )
        history = model.fit(train_dataset,
                    validation_data=valid_dataset
                    ,epochs=15,
                    verbose=1,
                    callbacks =[early_stop]
                    )
        model.save('artifacts/training/model.h5')
        return history