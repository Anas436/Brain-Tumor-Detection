import os 
import sys 
from pathlib import Path
import logging

#logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
filemanager =[
      "src/__init__.py",
      "src/components/__init__.py" ,
      "src/pipeline/__init__.py",
      "src/utils/__init__.py",
      "src/utils/common.py",
      "src/constant/__init__.py",
      "config/config.yaml",
      "params/params.yaml",
      "app.py",
        

]
for filepath in filemanager:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
      
        logging.info(f"{filename} is already exists")