import os
import  sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))
from src.pipeline.prediction import make_prediction

path = 'artifacts/data_ingestion/brain_tumor_dataset/no/8 no.jpg'
print(make_prediction(path))