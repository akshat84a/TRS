#!/usr/bin/env python@ajeet
# coding: utf-8

# Import requirements
import os
import cv2
import numpy as np

# Read images from directory
def read_images(directory_path):
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png')
    # Get subdirectories within the directory
    subdirectories = [subdir for subdir in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, subdir))]
    for subdir in subdirectories:
        subdirectory_path = os.path.join(directory_path, subdir)
        if not os.listdir(subdirectory_path):
            continue
        # Iterate over images in the subdirectory
        for filename in os.listdir(subdirectory_path):
            if filename.lower().endswith(valid_extensions):  
                image_path = os.path.join(subdirectory_path, filename)
                image = cv2.imread(image_path)
                # Resize the image to 30x30 pixels
                image = cv2.resize(image, (30, 30))
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                # Assign the label based on the subdirectory name
                label = int(subdir)
                labels.append(label)
    data = np.array(list(zip(images, labels)))  
    return data


# In[7]:

#train_data = read_images('/home/ba-ajeetkumary/Downloads/Ishan_Pr/data/Train')
#print(f"Train Data Shape : {train_data.shape}")






