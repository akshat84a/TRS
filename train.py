#!/usr/bin/env python
# coding: utf-8


#Import requirements
import numpy as np
from preprocess import read_images
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import model_maker
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load the dataset
train_data = read_images(r"C:\Users\Aksha\Desktop\College_Project\archive\Train")
shuffled_train = train_data.copy()  
shuffled_train = np.array(shuffled_train)  

# Split the shuffled_train array into training and testing sets
train_set, val_set = train_test_split(shuffled_train, test_size=0.3, random_state=42)

# Separate the input (X) and output/label (y) arrays from the training set
x_train = train_set[:, 0]
y_train = train_set[:, 1]

# Separate the input (X) and output/label (y) arrays from the testing set
x_val = val_set[:, 0]
y_val = val_set[:, 1]

# Convert the data type of the arrays to int
x_train = np.array([np.array(x) for x in x_train])
y_train = np.array([np.array(y) for y in y_train])
x_val = np.array([np.array(x) for x in x_val])
y_val = np.array([np.array(y) for y in y_val])

# Normalize the input data
x_train = x_train / 255.0
x_val = x_val / 255.0

# Convert target labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=43)
y_val = to_categorical(y_val, num_classes=43)

# Find the model
model = model_maker()
# Define learning rate
lr = 0.001
epochs = 1

opt = Adam(learning_rate=lr)  # Set the learning_rate instead of decay
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])



aug = ImageDataGenerator(
    rotation_range=15,            
    zoom_range=0.2,               
    width_shift_range=0.15,      
    height_shift_range=0.15,      
    shear_range=0.2,              
    horizontal_flip=True,         
    vertical_flip=True,           
    fill_mode="reflect"          
)


history = model.fit(aug.flow(x_train, y_train, batch_size=32), epochs=epochs, validation_data=(x_val,y_val))

model.save("krisna.h5")


# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()







