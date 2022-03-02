# import the necessary packages
import os
import argparse
import cv2

from PIL import Image
import numpy as np

import imutils
from imutils import paths


outpath = "/home/inthrustwetrust71/Desktop/jina_image_search_engine/data/flag_imgs/augmented_right"

path = "/home/inthrustwetrust71/Desktop/jina_image_search_engine/data/flag_imgs"

imagePaths = sorted(list(paths.list_images("./data/flag_imgs/right")))

for path in imagePaths:
    # print(path)
    print(path.split("/"))
    print()

    # Opening the image and converting 
    # it to RGB color mode
    # IMAGE_PATH => Path to the image
    img = Image.open(path).convert('RGB')
    
    # Extracting the image data &
    # creating an numpy array out of it
    img_arr = np.array(img)
    
    # Turning the pixel values of the 400x400 pixels to black 
    img_arr[0 : 100, 0 : 40] = (0, 0, 0)
    
    # Creating an image out of the previously modified array
    img = Image.fromarray(img_arr)
    
    # Displaying the image
    # img.show()

    # Save image
    # img.save(os.path.join(outpath, path.split("/")[-1]))
