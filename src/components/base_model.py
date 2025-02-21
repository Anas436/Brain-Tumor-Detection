import os 
import sys
from pathlib import Path
from dataclasses import dataclass
import tensorflow as tf 
from tensorflow.keras.layers import Dense,MaxPooling2D,BatchNormalization,Conv2D,Flatten,InputLayer,Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
from src.constant import CONFIG_FILE_PATH,PARAMS_FILE_PATH
from src.logger import logging
from src.utils.commonFunction import create_directories,read_yaml
from src.exception import CustomException


@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path:Path
    IMG_SIZE: list
    
class BaseModelConfigManager:
    
    def __init__(self,config_file_path=CONFIG_FILE_PATH,
                 params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_directories([self.config.artifacts_root])
        
    def get_base_model_config(self) -> BaseModelConfig:
        config =self.config.prepare_base_model
        create_directories([config.root_dir])
        
        base_model_config = BaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            IMG_SIZE = self.params.IMG_SIZE
        )
        return base_model_config
    
class BaseModel:
    
    def __init__(self,config:BaseModelConfig):
        self.config = config
    
    def model(self):
        try:
            self.lenet_model = tf.keras.Sequential([
                InputLayer(input_shape = (224,224,3)),
                Conv2D(filters=32,kernel_size=4,strides=1,activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2,2),strides=2),
                
                Conv2D(filters=32,kernel_size=4,strides=1,activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2,2),strides=2),
                
                Conv2D(filters=32,kernel_size=4,strides=1,activation='relu'),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2,2),strides=2),
                
                Flatten(),
                
                Dense(128,activation='relu'),
                BatchNormalization(),
                
                Dense(128,activation='relu'),
                BatchNormalization(),
                
                Dense(1,activation='sigmoid')
                
                ],name='Lenet_model')
            self.lenet_model.summary()
            self.lenet_model.save(self.config.base_model_path)
        except Exception as e:
            raise CustomException(e,sys)


    



