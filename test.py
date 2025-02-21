import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),)))
from src.utils.commonFunction import dataset_arrangement

train_dataset,valid_dataset,test_dataset = dataset_arrangement()