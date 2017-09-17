import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os

with open("sim_data_large.yaml", 'r') as stream:
    try:
        docs=yaml.load(stream)
        for doc in docs:
            for k,v in doc.items():
                if k == "annotations":
                    annotations = v  
                if k == "filename":
                    filename = v
            # open filename
            print(filename)
            base = os.path.basename(filename)
            img = cv2.imread(filename)
            img_size = img.shape
            print(img_size[0], img_size[1])
            
            # for each annotation in filename
            for i,annotation in enumerate(annotations):           
                # create region of interest (roi) based on xmin, x_width, y_min, y_width
                y1 = math.ceil(annotation["ymin"])
                y2 = y1 + math.ceil(annotation["y_height"])
                x1 = math.ceil(annotation["xmin"])
                x2 = x1 + math.ceil(annotation["x_width"])
                print(y1,y2, x1,x2)
                if (x1 > 0 and y1 > 0 and x2 < img_size[1] and y2 < img_size[0]):
                    roi = img[y1:y2, x1:x2]
                    cv2.imshow("cropped", roi)
                # save roi in the right label folder (green, red or yellow)
                label = annotation["class"]
                print(label)
                #use join so it works cross-platform
                output_filename = os.path.join("sim_data_capture",label, os.path.splitext(base)[0] + "_" + str(i) + os.path.splitext(base)[1])
                print(output_filename)
                cv2.imwrite(output_filename, roi)
               
    except yaml.YAMLError as exc:
        print(exc)