#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Import requirements

import matplotlib.pyplot as plt
from preprocess import read_images
import numpy as np


train_data = read_images('/home/ba-ajeetkumary/Downloads/Ishan_Pr/data/Train')
# Label Overview
classes = { 
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons'
}
# Convert label values to integers
labels = train_data[:, 1].astype(int)

# Get the unique labels and their counts
unique_labels, label_counts = np.unique(labels, return_counts=True)

# Create a horizontal bar plot
plt.figure(figsize=(15,10))
plt.barh(unique_labels, label_counts)
plt.xlabel('Count')
plt.ylabel('Label')
plt.title('Label Counts in Train Data')
plt.yticks(unique_labels, [classes[label] for label in unique_labels])

# Select 10 random samples from train_data
indices = np.random.choice(train_data.shape[0], size=10, replace=False)
samples = train_data[indices]
# Create a grid of subplots for the images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.subplots_adjust(hspace=0.4)

for i, (image, label) in enumerate(samples):
    ax = axes[i // 5, i % 5]
    ax.imshow(image)
    ax.set_title(classes[label])
    ax.axis('off')

plt.show()







