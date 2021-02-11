##  FlowerWiki

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from random import shuffle
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,Activation,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model
from keras.applications.vgg16 import decode_predictions
from keras.models import load_model
from keras import backend as K



## taking inputs
IMG_SIZE = 128

my_new_model = Sequential()
my_new_model.add(VGG16(include_top=False,
                       pooling='avg',
                       weights='imagenet'))
my_new_model.add(Dense(5,activation='softmax'))

my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

my_new_model.summary()

datagen = ImageDataGenerator(rescale=1./255,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=90,
                             horizontal_flip=True)


val_gen = datagen.flow_from_directory('Compressed/val_data',
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=20,
        class_mode='categorical')

train_gen = datagen.flow_from_directory(
        'Compressed/Flowers',
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=10,
        class_mode='categorical')

test = my_new_model.fit_generator(train_gen,               ### for transfer learning
                    epochs=5,
                    steps_per_epoch=len(train_gen),
                    validation_steps=len(val_gen))



######### plotting graphs ################
import matplotlib.pyplot as plt
## loss
plt.plot(test.history['acc'],label = 'train loss')
plt.plot(test.history['loss'],label = 'val loss')
plt.legend()
plt.show()


my_new_model.save(r'F:\data analytics and ML\flower recognition\my_model1.sav')


## training model