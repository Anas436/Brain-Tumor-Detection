import os 
import sys
import tensorflow as tf 
import cv2
import numpy as  np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import onnxruntime as nxrun
import numpy as np
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../../')))
from src.utils.common import  dataset_arrangement
from src.logger import logging
from src.exception import CustomException

train_dataset,valid_dataset,test_dataset = dataset_arrangement()

# from path to prediction 
def make_prediction(path):
    class_names = ['no','yes']
    model = load_model(os.path.join('model','model.h5'))
    image = load_img(path,target_size=(224,224))

    img = img_to_array(image)
    print(img.shape)
    img = tf.expand_dims(img,axis=0)
    pred = np.array(tf.argmax(model.predict(img),axis=-1))[0]
    result =class_names[pred]
    print(result)
    if result == 'no':
        return str('You are free from brain tumor!!!')
    else:
        return str('Yes you have brain tumor.You should go to hospital!!!')
# for app
# from image to make prediction  
def create_prediction(image,target_size=(224,224)):
    class_names = ['no','yes']
    model = load_model(os.path.join('model','model.h5'))
    image = image.resize(target_size)

    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    pred = np.array(np.argmax(model.predict(image),axis=-1))[0]
    result =class_names[pred]

    if result == 'yes':
        return str('Yes, you have brain tumor.You should go to hospital!!!')
    else:
        return str('You are free from brain tumor!!!')
    

def onnxModel_prediction(image,target_size=(224,224)):
     class_names = ['no','yes']
     image = image.resize(target_size)
     image = np.float32(image)
     ximg = np.expand_dims(image,0)

     sess = nxrun.InferenceSession("model/model.onnx")


     input_name = sess.get_inputs()[0].name
     label_name = sess.get_outputs()[0].name
     result = sess.run([label_name], {input_name: ximg})
     prob = result[0]
     pred = np.argmax(prob,-1)[0]
     result =class_names[pred]

     if result == 'yes':
        return str('Yes, you have brain tumor.You should go to hospital!!!')
     else:
        return str('You are free from brain tumor!!!')

