from flask import Flask
#from keras.models import load_model
from flask import request
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
from keras.models import load_model


import cv2

app = Flask(__name__)

path = "my_model.h5"
model = load_model(path)

size=(128, 128)
#global model
#model = ResNet50(weights="imagenet")
#TODO: Load model file

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def predict():
  im = request.files.get('image')
  #file = open('testfile','w')
  im.save('testfile')
  #file.write(im)
  print(im)
  img = cv2.resize(cv2.imread('testfile'), size)
  result = model.predict(np.array([img]))
  #print(image = request.files['image'].read())

  #image = request.files['file']
  #image = Image.open(image)
  # transform
  #result = model.predict(np.array([image]))
  return np.array_str(result)
  #return "success"

app.run(host='0.0.0.0', port=5000)
