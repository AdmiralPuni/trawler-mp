import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import json
import numpy as np
import keras
import tensorflow.keras.preprocessing.image as image
import numpy as np
from PIL import Image

settings_json = json.load(open('settings.json', 'r'))

character_names = settings_json['models']['myusu']['names']
model_filename = settings_json['models']['myusu']['filename']


#loading model from hdf5 file
def load_model(character_count, model_filename):
  model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(character_count, activation='softmax')
  ])

  #load saved model
  model.load_weights("models/" + model_filename)
  model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model

print('Loading model...')
model = load_model(len(character_names), model_filename)

def get_name(img):
  img = img.resize((150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, verbose=0)

  largest = 0

  #Get the largest class, workaround for numbers with e notation
  for x in range(0, len(classes[0])):
    if(classes[0][x] > largest):
      largest = x
  
  return character_names[largest]

def load_img(path):
  img = Image.open(path)
  img = img.convert('RGB')
  return img

#print(get_name(load_img('374.png')))