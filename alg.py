import pandas as pd
import numpy as np
import tensorflow.keras as keras
import os
import matplotlib.pyplot as plt
import seaborn as sb
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,MaxPool2D,Conv2D,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input as inputse
from log import logging
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
import pickle as pkl

# Initially clean,transform,analyze the input and make it as a function so that same can be used for testing and new data prediction.

class Data_Analysis:
    def __init__(self):
        self.IMAGE_Size = 150

    def clean(self,images):
        # To load the images and convert into numpy array format.
        self.input_images = images
        try:
            self.array_values = cv2.imread(self.input_images)  # reading the image and converting it into array
            self.array_values = cv2.resize(self.array_values, (self.IMAGE_Size, self.IMAGE_Size))
            return self.array_values
        except Exception as e:
            print('Exception occured ,{}'.format(str(e)))
            logging.error("error during image loading"+str(e))
            return None

    def convert(self,array_images):
        # To apply convert like dividing with 255
        try:
            self.array_images=np.array(array_images)/255
            return self.array_images
        except Exception as e:
            print('Exception Occured ,{}'.format(str(e)))
            logging.error("error during numpy calculations ,"+str(e))
            return None

    def dependent(self,labels):
        try:
            labels_encoder=LabelEncoder()
            labels_rencoded=labels_encoder.fit_transform(labels)
            pkl.dump(labels_encoder,open('labencdr.pkl','wb'))
            print(labels_encoder.inverse_transform(labels_rencoded))
            labels_en_category=to_categorical(labels_rencoded)
            return labels_en_category
        except Exception as e:
            print('Exception Occured ,{}'.format(str(e)))
            logging.error('error during dependent variable transformation. ,{}'.format(e))
            return None

class algor:
    # def __init__(self):
    #     self.algorithm_value=Sequential()


    def algorithm(self):
        cnv=Sequential()
        vgg_16 = VGG16(include_top=False,
                       input_shape=(150, 150, 3),
                       weights='imagenet')
        for layers in vgg_16.layers:
            layers.trainable = False  # Freezing the model to not to train and update the existing weights.
            cnv.add(layers)
        # include_top=False won't include the last three neural network layers only till flattened layer will be present.
        # input_shape to be mentioned when include_top=False given.
        cnv.add(Flatten())
        cnv.add(Dense(units=64, activation='relu', kernel_initializer='glorot_uniform'))
        cnv.add(Dense(units=5, activation='softmax'))
        cnv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return cnv

    def transform(self):
        # To apply transformations like augmentation.
        gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True)
        return gen

daisy_dir='images/daisy/'
dandelion_dir='images/dandelion/'
rose_dir='images/rose/'
sunflower_dir='images/sunflower/'
tulip_dir='images/tulip/'


all_dir={daisy_dir:'daisy',dandelion_dir:'dandelion',rose_dir:'rose',sunflower_dir:'sunflower',tulip_dir:'tulip'}
Var=[]
labels=[]

obj_da=Data_Analysis()

for dir,value in all_dir.items():
    for name in os.listdir(dir):
        array_values=obj_da.clean(dir+name)
        Var.append(array_values)
        labels.append(value)
print(set(labels))
logging.info('Input array created.')
print(len(Var))
print(len(labels))
print(np.array(Var).shape)
array_var=obj_da.convert(Var)
labels_roencoded=obj_da.dependent(labels)
print(array_var.shape)
x_train,x_test,y_train,y_test=train_test_split(array_var,labels_roencoded,test_size=0.29)
obj=algor()
algorithm=obj.algorithm()
agm=obj.transform()
#agm.fit(x_train)
#algorithm.fit(agm.flow(x_train,y_train),validation_data=(x_test,y_test),epochs=47)
#algorithm.save('alg.h5')