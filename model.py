#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Import requirements
import tensorflow as tf            
from tensorflow import keras

# Build the model
def model_maker():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(30,30,3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(axis=-1),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(axis=-1),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.5),

        tf.keras.layers.Dense(43, activation='softmax')
    ])
    return model


# In[13]:


if __name__=='__main__':
    model_maker()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




