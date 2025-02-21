import os 
import  sys
import tensorflow as tf 
from tensorflow.keras.models import load_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.components.model_trainer import trainer
from src.logger import logging
from src.exception import CustomException
from src.utils.commonFunction import dataset_arrangement

    
#data arangement
try:
    train_dataset,valid_dataset,test_dataset = dataset_arrangement()
except Exception as e:
    raise CustomException(e,sys)

# train the model
try:
    train = trainer()
    train.get_train_model()
except Exception as e :
    raise CustomException(e,sys)