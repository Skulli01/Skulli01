#import libraries

import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from tensorflow.keras.optimizers import SGD,Adam

import tensorflow as tf

import cv2
import os
import pandas as pd
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#loading data

#chosen image size
imgSize = 71
#possible labels
labels = ['vehicles', 'non-vehicles']
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        index = labels.index(label)
        for image in os.listdir(path):
            img = cv2.imread(os.path.join(path, image)) 
            #convert BGR to RGB format
            imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #error check
            if img.any() == None:
                print("Error in loading files")
            #reshaping the images to standard size of imgSize x imgSize
            resizedImg = cv2.resize(imgRgb, (imgSize, imgSize))
            #printing out the images:
            #cv2.imshow('window',resizedImg)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            data.append([resizedImg, index])
    
    return np.array(data, dtype=object)


data = get_data('/home/oma1/scratch/data') #Loading the data

#using pandas dataframe to split the data into training and testing sets
df = pd.DataFrame(data,columns=['image','label'])
X_train, X_test, y_train, y_test = train_test_split(df['image'], df['label'], test_size=0.2, random_state=42)


#reshaping and normalising the training and testing sets

X_train=np.reshape(X_train.to_list(),(len(X_train),imgSize,imgSize,3))
X_test=np.reshape(X_test.to_list(),(len(X_test),imgSize,imgSize,3))
X_train=X_train/255.0
X_test=X_test/255.0
X_label=np.array(y_train,dtype=int)
test_label=np.array(y_test,dtype=int)
y_test=np.array(y_test,dtype=int)


#generate batches of tensor image data with real-time data augmentation.

generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

#fitting onto training data set to produce more training tensor image datas

#creating the layers of the neural network

from keras.applications.xception import Xception

baseModel = Xception(weights='imagenet', input_shape = (imgSize,imgSize,3), include_top=False)
baseModel.trainable = False
Tmodel = Sequential()
Tmodel.add(baseModel)
Tmodel.add(GlobalAveragePooling2D())
Tmodel.add(Dense(140))
Tmodel.add(Dropout(0.2))
Tmodel.add(Dense(1, activation = 'sigmoid'))
Tmodel.summary()
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

#copy seperate models for distinct optimsers
TmodelAdam = Tmodel
TmodelSGD = Tmodel

#result of our model using SGD as optimiser
TmodelSGD.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
resultSGD = TmodelSGD.fit(X_train,X_label,epochs = 5, validation_data = (X_test, test_label), callbacks = [callbacks], verbose = 1)

#plotting training accuracy and testing accuracy against epochs using history callback

acc = resultSGD.history['accuracy']
test_acc = resultSGD.history['val_accuracy']
loss = resultSGD.history['loss']
test_loss = resultSGD.history['val_loss']
epochs_range = range(5)
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Testing Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Testing Accuracy')
plt.show()

#result of our model using Adam as optimiser

optiAdam = 'Adam'
TmodelAdam.compile(optimizer=optiAdam, loss='binary_crossentropy', metrics=['accuracy'])
resultAdam = TmodelAdam.fit(X_train,X_label,epochs = 5, validation_data = (X_test, test_label), callbacks = [callbacks], verbose = 1)

#plotting training accuracy and testing accuracy against epochs using history callback

acc = resultSGD.history['accuracy']
test_acc = resultSGD.history['val_accuracy']
loss = resultSGD.history['loss']
test_loss = resultSGD.history['val_loss']
epochs_range = range(5)
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Testing Accuracy')
plt.show()
