#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 19:00:49 2017

@author: kyleguan
"""
# Note: some imported package are not used in this script. But they will be handy
# when we improve the model.
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import sys, os, time
from datetime import timedelta
import pickle
import itertools
import math, random
import glob

import cv2
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import initializers



# The paths to the image dataset and trained models
base_image_path = "5_tensorflow_traffic_light_images/"
light_colors = ["red", "green", "yellow"]
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'tl_model_2.h5'


# Resize the images to the size of 32X32 and save the in the list named data
data=[]
color_counts = np.zeros(3)
for color in light_colors:
    for img_file in glob.glob(os.path.join(base_image_path, color, "*")):
        img = cv2.imread(img_file)
        if not img is None:
            img = cv2.resize(img, (32, 32))
            label = light_colors.index(color)
            assert(img.shape == (32, 32, 3))
            data.append((img, label, img_file))
            color_counts[light_colors.index(color)]+=1
                         
#Divide the data into training, validation, and test sets.                         
random.shuffle(data)
X, y, files = [], [],[]
for sample in data:
    X.append(sample[0])
    y.append(sample[1])
    files.append(sample[2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.086,
    random_state = 832289)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.094,
    random_state = 832289)
print("No. of training samples: %d, No. of test samples: %d, No. of validation samples: %d"\
%(len(X_train), len(X_test), len(X_valid)) )    

# Data preprocessing: converting to numpy array, normalizing data, and creating
# one-hot labels.
X_train=np.array(X_train)
X_valid=np.array(X_valid)
X_test=np.array(X_test)
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
encoder = LabelBinarizer()
encoder.fit(y_train)
X_train /= 255
X_valid /= 255
X_test /= 255
y_train_onehot = encoder.transform(y_train)
y_valid_onehot = encoder.transform(y_valid)
y_test_onehot = encoder.transform(y_test)

batch_size = 32
num_classes = 3
epochs = 25 #200

# Building a simple CNN model
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

# Optimization and training
opt=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.fit(X_train, y_train_onehot,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_valid, y_valid_onehot),
              shuffle=True)
# Save trained model
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# Evaluate the trained model
evaluation = model.evaluate(X_test, y_test_onehot, batch_size = batch_size, verbose=0)
print('Model Accuracy = %.2f' % (evaluation[1]))
predict = model.predict(X_test, batch_size = batch_size)

# Plot an example
plt.imshow(cv2.cvtColor(X_test[0], cv2.COLOR_BGR2RGB))
plt.title('Label: '+str(y_test[0]) + ', Predict: '+str(np.argmax(predict[0])))






                    