import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageFont
from PIL import ImageDraw
import classifier

print("Loading MobileNet...")
module_handle = "models/mobilenet/"
detector = hub.load(module_handle).signatures['default']

color = ImageColor.getrgb('#6AE670')
font = None
try:
  font = ImageFont.truetype("arial.ttf", 20)
except:
  try:
    font = ImageFont.truetype("/usr/share/fonts/noto/NotoSans-Regular.ttf", 20)
  except:
    font = ImageFont.load_default()


#Draw boxes and text on image, I don't know how this works
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
  draw.line([(left, top),(right, bottom)], width=thickness, fill=color)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
    draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
    text_bottom -= text_height - 2 * margin

def crop_image(image, ymin, xmin, ymax, xmax):
  image_pil = image
  im_width, im_height = image_pil.size
  (left, top, right, bottom) = (xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height)
  image_pil = image_pil.crop((left, top, right, bottom))
  return image_pil

def draw_boxes(loaded_image, boxes, class_names, scores, max_boxes=10, min_score=0.3):
  #decode class_names to ascii
  for i in range(len(class_names)):
    class_names[i] = class_names[i].decode("ascii")
  
  if "Human face" not in class_names:
    return loaded_image, ['NULL']


  color = ImageColor.getrgb('#6AE670')
  try:
    font = ImageFont.truetype("arial.ttf", 20)
  except:
    try:
      font = ImageFont.truetype("/usr/share/fonts/noto/NotoSans-Regular.ttf", 20)
    except:
      font = ImageFont.load_default()
  
  image_boxes = loaded_image
  detected_names = class_names

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      if class_names[i] == "Human face":
        ymin, xmin, ymax, xmax = tuple(boxes[i])

        image_pil = Image.fromarray(np.uint8(loaded_image)).convert("RGB")
        cropped_image = crop_image(image_pil, ymin, xmin, ymax, xmax)
        name = classifier.get_name(cropped_image)
        class_names[i] = name

        display_str = "{}".format(name)
        
        draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
        
        
        np.copyto(image_boxes, np.array(image_pil))
      else:
        ymin, xmin, ymax, xmax = tuple(boxes[i])

        image_pil = Image.fromarray(np.uint8(loaded_image)).convert("RGB")
        cropped_image = crop_image(image_pil, ymin, xmin, ymax, xmax)

        display_str = "{}".format(class_names[i])
        
        draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
        
        
        np.copyto(image_boxes, np.array(image_pil))
  
  return image_boxes, class_names

def draw_all_boxes(loaded_image, boxes, class_names, scores, max_boxes=10, min_score=0.3):
  global font
  global color
  #decode class_names to ascii
  for i in range(len(class_names)):
    class_names[i] = class_names[i].decode("ascii")
  

  image_boxes = loaded_image
  image_pil = Image.fromarray(np.uint8(loaded_image)).convert("RGB")

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])

      display_str = "{}".format(class_names[i])
      
      draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
      
      np.copyto(image_boxes, np.array(image_pil))
  
  return image_boxes, class_names

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector_fromfile(path):
  global detector
  img = load_img(path)
  detected_names = []

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  result = detector(converted_img)

  result = {key:value.numpy() for key,value in result.items()}

  #detect faces>draw box around them> detect the face<repeat >return image with box and detected names
  image_with_boxes, detected_names = draw_boxes( img.numpy(), result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])

  Image.fromarray(np.uint8(image_with_boxes)).convert("RGB").save('output.png')

def run_detector(img):
  global detector
  detected_names = []

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  result = detector(converted_img)

  result = {key:value.numpy() for key,value in result.items()}

  #detect faces>draw box around them> detect the face<repeat >return image with box and detected names
  image_with_boxes, detected_names = draw_all_boxes( img, result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])
  #image_with_boxes, detected_names = draw_boxes( img, result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])

  return image_with_boxes, detected_names

#run_detector_fromfile('test.png')