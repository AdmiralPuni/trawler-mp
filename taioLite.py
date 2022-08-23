import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import getopt
import json
import gc

import tkinter as tk
from tkinter.constants import BOTH, LEFT, N, NO, TOP, X, YES

import numpy as np
import tensorflow.keras as keras
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np
from PIL import ImageTk
from PIL import Image
from shutil import move

settings_json = json.load(open('settings.json', 'r'))


#loading model from hdf5 file
def compile_saved_model(character_count, model_filename):
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

def print_help():
  print('Example     : taio.py -i input -o output -m myusu -s')
  print('Arguments   :')
  print('-i | Input folder, place your unsorted image here')
  print('-o | Output folder, sorted images will be moved here')
  print('-m | Model for character face detection')
  print('-l | List the models available')
  print('-s | Supervised decision, you will decide for each image')
  print('-a | Images will be automaticcaly moved to their folder WARNING: Not very accurate')
  print('-h | Show this help menu')

def extract_arguments(argv):
  #Get launch argument
  #Todo: load large imports after argument check instead of before
  try:
    opts, args = getopt.getopt(argv, "hli:o:m:sa", ["i=", "o=", "m="])
  except getopt.GetoptError:
    print_help()
    sys.exit(2)
  if len(opts) == 0:
    print_help()
    sys.exit()
  for opt, arg in opts:
    if opt == '-h':
      print_help()
      sys.exit()
    elif opt in ("-i"):
      input_directory = arg
    elif opt in ("-o"):
      output_directory = arg
    elif opt in ("-m"):
      selected_model = arg
    elif opt in ("-s"):
      supervision = True
    elif opt in ("-a"):
      supervision = False
  
  return input_directory, output_directory, selected_model, supervision
  
def move_file(path, detected_names, single, output_directory, selected_model):
  sorted_folder = output_directory + "/" + selected_model + '/'
  if single:
    if not os.path.exists(sorted_folder + detected_names[0]):
      os.makedirs(sorted_folder + detected_names[0])
    move(path, sorted_folder + detected_names[0] + "/" + os.path.basename(path))
  else:
    #WARNING: If the character name has - it will be hard to read the folder
    if not os.path.exists(sorted_folder + '-'.join(detected_names)):
      os.makedirs(sorted_folder + '-'.join(detected_names))
    move(path, sorted_folder + '-'.join(detected_names) + "/" + os.path.basename(path))

def clear_memory():
  #Attempting to free up RAM, hub ram in still in shackles
  #Todo: Doesn't work on linux
  keras.backend.clear_session()
  gc.collect()

def load_model(model_name):
  character_names = settings_json['models'][model_name]['names']
  model_filename = settings_json['models'][model_name]['filename']
  model = compile_saved_model(len(character_names), model_filename)
  return character_names, model

def detect(model, image_path, character_names, return_names = True):
  img = image.load_img(image_path, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)

  if return_names:
    largest = 0
    for x in range(0, len(classes[0])):
      if(classes[0][x] > largest):
        largest = x
    return character_names[largest]
  else:
    return classes

class prediction:
  def __init__(self, path, names, image):
    self.path = path
    self.names = names
    self.image = image

def run_auto(input_directory, output_directory, selected_model, override_character = None):
  prediction_list = []
  
  if override_character != None:
    character_names = override_character
    model = compile_saved_model(len(override_character), selected_model + '.hdf5')
  else:
    character_names, model = load_model(selected_model)

  for filename in tqdm(os.listdir(input_directory)):
    name_fix = []
    if filename.endswith(".jpg") or filename.endswith(".png"):
      path = os.path.join(input_directory, filename)
      name_fix.append(detect(model, path, character_names, return_names = True))
      prediction_list.append(prediction(path, name_fix, path))
    else:
        continue

  clear_memory()

  for list in tqdm(prediction_list):
    if len(list.names) != 0:
      move_file(list.path, list.names, True, output_directory, selected_model)

def main(argv):
  input_directory, output_directory, selected_model, supervision = extract_arguments(argv)
  prediction_list = []

  #Class to save prediction result
  class prediction:
    def __init__(self, path, names, image):
      self.path = path
      self.names = names
      self.image = image

  print('Input Directory     :', input_directory)
  print('Output Directory    :', output_directory)
  print('Model               :', selected_model)
  print('Supervision         :', supervision)

  #Load character names and the model filename from settings.json
  character_names, model = load_model(selected_model)

  #Iterates through images in input folder
  print("Running detections...")
  for filename in tqdm(os.listdir(input_directory)):
    name_fix = []
    if filename.endswith(".jpg") or filename.endswith(".png"):
      path = os.path.join(input_directory, filename)
      name_fix.append(detect(model, path, character_names, return_names = True))
      prediction_list.append(prediction(path, name_fix, path))
    else:
        continue

  clear_memory()

  def auto():
    print('Moving files...')
    for list in tqdm(prediction_list):
      if len(list.names) != 0:
        move_file(list.path, list.names, True, output_directory, selected_model)
  
  #Will most likely stays like this unless GUI change is really needed
  def gui():
    root = tk.Tk()

    root.title("Tensorflow Assisted Image Organizer | TAIO v0.1")

    def detection_count(detection_correct ,true_count = [], false_count = []):
      if detection_correct:
        true_count.append(0)
        return len(true_count)
      else:
        false_count.append(0)
        return len(false_count)

    #Cycles through the prediction list
    def cycle_prediction(index, single, save=True):
      
      #move the predicted image to the folder of the name
      current_prediction = prediction_list[index]
      if save:
        detection_count(True)
        move_file(current_prediction.path, current_prediction.names, single, output_directory, selected_model)
      else:
        detection_count(False)
      #attempting to treat out of index on last prediction
      try:
        #Load the next prediction to show in the window
        current_prediction = prediction_list[index+1]
        #Skip if prediction doesn't detect any names
        if len(current_prediction.names) == 0:
          cycle_prediction(get_index(), False, False)
          return
        change_pic(vlabel, current_prediction.image)
        change_name(detection_text, '-'.join(current_prediction.names))
        change_name(label_path, current_prediction.path)
      except:
        change_name(detection_text, 'Task completed')
        false_count = detection_count(False)-1
        true_count = detection_count(True)-1
        detection_accuracy = (true_count/(false_count + true_count))*100
        change_name(label_path, 'FC=' + str(false_count) + ' TC=' + str(true_count) + ' ACC=' + str(round(detection_accuracy,2)) + '%')

    #Iterates when called
    def get_index(index = []):
      index.append(0)
      return len(index)-1

    #Change the picture in the window
    def change_pic(labelname, file_path, max_width = 500):
        loaded_image = Image.open(file_path)
        width, height = loaded_image.size
        ratio_height = height/width
        photo1 = ImageTk.PhotoImage(loaded_image.resize((max_width, round(max_width*ratio_height))))
        labelname.configure(image=photo1)
        labelname.photo = photo1

    #Change the name in the window
    def change_name(labelname, text):
        labelname.configure(text=text)

    #Load buttons
    fm = tk.Frame(root)
    button_single = tk.Button(fm, text="Single", width=15,command=lambda: cycle_prediction(get_index(), True), font="Calibri 20")
    button_false = tk.Button(fm, text="False", width=15,command=lambda: cycle_prediction(get_index(), True, False), font="Calibri 20")
    button_single.pack(side=LEFT, anchor=N, fill=X, expand=YES)
    button_false.pack(side=LEFT, anchor=N, fill=X, expand=YES)
    fm.pack(side=TOP, fill=BOTH, expand=NO)

    #Load label for detected names
    frame_main = tk.Frame(root)
    vlabel = tk.Label(frame_main)
    photo = ImageTk.PhotoImage(Image.open(output_directory + "/temp.png").resize((500,500)))
    vlabel.configure(image=photo)
    detection_text = tk.Label(frame_main, text="Detected names", font="Calibri 24")
    label_path = tk.Label(frame_main, font="Calibri 16")
    detection_text.pack(side=TOP, anchor=N)
    label_path.pack(side=TOP, fill=BOTH, expand=NO)
    vlabel.pack(side=TOP, anchor=N)
    frame_main.pack(side=TOP, fill=BOTH, expand=YES)

    #Show the first item in the prediction list
    #Todo: Ignore the first detection if the name is empty
    current_prediction = prediction_list[0]
    change_pic(vlabel, current_prediction.image)
    change_name(detection_text, '-'.join(current_prediction.names))
    change_name(label_path, current_prediction.path)

    root.mainloop()
  
  if supervision:
    gui()
  else:
    auto()


if __name__ == "__main__":
  main(sys.argv[1:])
