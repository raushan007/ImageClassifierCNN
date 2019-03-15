# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 20:07:39 2019

@author: ASUS
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


# initializing the neural network
classifier = Sequential()

#Adding first convolutional network
classifier.add(Convolution2D(32,(3,3),input_shape =(64,64,3),activation='relu'))

#Adding first Max Pooling Layers
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding second convolutional layers
classifier.add(Convolution2D(64,(3,3),activation='relu'))

#Adding Second Max pooling Layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Now flattening the max pooling layer to feed into Artificial Neural Network
classifier.add(Flatten())

# Full connected Articial neural Network
classifier.add(Dense(units =128 ,activation='relu'))
classifier.add(Dense(units =1 ,activation='sigmoid'))

#compiling the CNN network
classifier.compile(optimizer='adam' ,loss='binary_crossentropy',metrics=['accuracy'])

#Now preprocessing the input image
#Now preprocessing the input image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2 ,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64,64),
                                                 batch_size =32 ,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64,64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         epochs = 25 ,
                         validation_data = test_set,
                         validation_steps =2000)

import numpy as np
from keras.preprocessing import image

#loading the image and then changing the image into array so it becomes 3D array as it contains RGB color
test_image = image.load_img('single_prediction/cat_or_dog_2.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)

# But it takes 4D array as input because 4 dimension is for batch size which is equal to 1
test_image = np.expand_dims(test_image,axis = 0)

#Predicting the value of the image
result = classifier.predict(test_image)

training_set.class_indices  #it will tell that 1 belongs to dog or cat
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)


classifier.save('trained_model.h5')

from keras.models import load_model
new_model = load_model('trained_model.h5')

new_model.summary()



