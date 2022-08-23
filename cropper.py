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

def crop_image(output, image, ymin, xmin, ymax, xmax):
  image_pil = image
  im_width, im_height = image_pil.size
  (left, top, right, bottom) = (xmin * im_width,ymin * im_height, xmax * im_width, ymax * im_height)
  image_pil = image_pil.crop((left, top, right, bottom))
  image_pil.save(output)
  #image_pil.save("/content/crop.png")

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

def draw_boxes(output, image, boxes, class_names, scores, max_boxes=10, min_score=0.3):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      if class_names[i].decode("ascii") == "Human face" or class_names[i].decode("ascii") == "Human head":
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        
        display_str = "{}: {}%".format(class_names[i].decode("ascii"), int(100 * scores[i]))
        color = colors[hash(class_names[i]) % len(colors)]
        image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
        cropped_image = crop_image(image_pil, ymin, xmin, ymax, xmax)
        print(classifier.get_name(cropped_image))

        draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
  return image


#module_handle = "models/rcnn/"
module_handle = "models/mobilenet/"
detector = hub.load(module_handle).signatures['default']

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def run_detector(path, output):
  global detector
  try:
    img = load_img(path)
  except:
    return

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  result = detector(converted_img)

  result = {key:value.numpy() for key,value in result.items()}

  image_with_boxes = draw_boxes(output, img.numpy(), result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])
  Image.fromarray(image_with_boxes).save(output)

run_detector('test.png', 'output.png')