import os 
import  sys
import tensorflow as tf 
from tensorflow.keras.models import load_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.components.model_trainer import trainer
from src.logger import logging
from src.exception import CustomException

from src.components.base_model import BaseModelConfigManager,BaseModel,BaseModelConfig
from src.components.data_load import DataingestionConfigManager,Dataload,DataingestionConfig


try:
    from src.utils.commonFunction import dataset_arrangement
    train_dataset,valid_dataset,test_dataset = dataset_arrangement()
except Exception as e:
    raise CustomException(e,sys)
# base model
try:
    base_model_config = BaseModelConfigManager()
    config = base_model_config.get_base_model_config()
    base_model = BaseModel(config)
    base_model.model()
except Exception as e:
    raise CustomException(e,sys)

# train the model
try:
    train = trainer()
    train.get_train_model()
except Exception as e :
    raise CustomException(e,sys)