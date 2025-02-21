import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))
from  src.components.data_load import DataingestionConfigManager,Dataload,DataingestionConfig
from src.components.data_ingestion import DataIngestion

def download_dataset():
    data_ingestion_config = DataingestionConfigManager()
    data_ingestion_config = data_ingestion_config.get_data_config_manager()
    data_load = Dataload(data_ingestion_config)
    data_load.download_dataset()
    data_load.Unzip_dataset()

def dataset_arrangement():
    download_dataset()
    #load the data
    data_ingestion = DataIngestion() 
    dataset = data_ingestion.load_dataset_from_dir(directory= os.path.join("artifacts", "data_ingestion", "brain_tumor_dataset"))
    train_dataset,valid_dataset,test_dataset=data_ingestion.split_dataset(dataset=dataset,
                                                                        Train_ratio=0.7,
                                                                        Val_ratio=0.2,
                                                                        Test_ratio=0.1
                                                                        )
    train_dataset,valid_dataset,test_dataset =data_ingestion.partition_dataset(
                                                                        train_dataset,
                                                                        valid_dataset,
                                                                        test_dataset
                                                                        )
    return train_dataset,valid_dataset,test_dataset