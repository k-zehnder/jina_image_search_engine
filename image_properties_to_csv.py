from genericpath import exists
import os
import numpy
import cv2
import argparse
import numpy
import csv

from imutils import paths

my_file="./data/pqdata/my_details.csv"
data_path = "./data/pqdata/data"

# check if file exists 
if os.path.exists(my_file):
    os.remove(my_file)
    print("The file: {} is deleted!".format(my_file))
else:
    print("The file: {} does not exist!".format(my_file))


with open(my_file, 'w', newline = '') as file:
    writer = csv.writer(file)
        
    writer.writerow(["S.No.", "Name", "Country", "Type", "Height",
                        "Width", "Channels",
                        "Avg Blue", "Avg Red",
                        "Avg Green"])
  
imagePaths = sorted(list(paths.list_images(data_path)))
for idx, path in enumerate(imagePaths):
    print(path)
    country = path.split("/")[-2]
    type = "flag"
    
    try:
        img = cv2.imread(path)
        h, w, c = img.shape
        print(h, w, c)

        avg_color_per_row = numpy.average(img, axis = 0)
        avg_color = numpy.average(avg_color_per_row, axis = 0)
      
        with open(my_file, 'a', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow([idx, path, country, type, h, w, c, 
                            avg_color[0], avg_color[1],
                            avg_color[2]])
            file.close()

    except Exception as e:
        print("[INFO] problem reading image from disk")
        print(e)

    print()
    print()
