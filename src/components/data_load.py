import tensorflow as tf 
import os 
import sys
from dataclasses import dataclass
from pathlib import  Path
import gdown
import zipfile
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))
from src.utils.commonFunction import read_yaml,create_directories
from src.constant import CONFIG_FILE_PATH,PARAMS_FILE_PATH
from src.logger import logging
from src.exception import CustomException

@dataclass
class DataingestionConfig:
    root_dir: Path
    source_url: str
    zip_path: Path
    unzip_path: Path
    

class DataingestionConfigManager:
    
    def __init__(self,cofig_file_path=CONFIG_FILE_PATH,
                 params_file_path=PARAMS_FILE_PATH):
        self.config = read_yaml(cofig_file_path)
        self.params = read_yaml(params_file_path)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_config_manager(self) ->DataingestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        ingestion_manager = DataingestionConfig(
            root_dir = Path(config.root_dir),
            source_url= str(config.source_url),
            zip_path = Path(config.zip_path),
            unzip_path = Path(config.unzip_path)
        )
        return ingestion_manager
    

class Dataload:
    
    def __init__(self,config: DataingestionConfig):
        self.config = config
        
    def download_dataset(self):
       try: 
            source_url = self.config.source_url
            zip_dir = self.config.zip_path
            os.makedirs("artifacts/data_ingestion",exist_ok=True)
            
            id_name = source_url.split('/')[-2]
            prefix =  'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+id_name,str(zip_dir))
       except Exception as e:
           raise CustomException(e,sys)
    def Unzip_dataset(self):
        zip_dir = self.config.zip_path
        unzip_dir = self.config.unzip_path
        os.makedirs(unzip_dir,exist_ok=True)
        with zipfile.ZipFile(zip_dir,'r') as zip_obj:
            zip_obj.extractall(unzip_dir)
            

if __name__ == '__main__':
    try:
        data_ingestion_manager = DataingestionConfigManager()
        data_ingestion_config = Dataload(config=data_ingestion_manager.get_data_config_manager())
        data_ingestion_config.download_dataset()
        data_ingestion_config.Unzip_dataset()
    except Exception as e:
        raise CustomException(e,sys)