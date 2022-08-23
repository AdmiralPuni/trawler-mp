import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
import livetest


bounding_box = {'top': 200, 'left': 400, 'width': 1280, 'height': 720}

sct = mss()

while True:
  sct_img = sct.grab(bounding_box)
  #conver sct_img to pillow image
  img = Image.frombytes("RGB", sct_img.size, sct_img.rgb, 'raw', "BGR")
  result, names = livetest.run_detector(np.array(img))
  print(names)

  #cv2.imshow('screen', np.array(img))
  cv2.imshow('screen', result)
  
  #time.sleep(0.3)

  if (cv2.waitKey(1) & 0xFF) == ord('q'):
    cv2.destroyAllWindows()
    break