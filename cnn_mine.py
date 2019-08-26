#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 09:58:31 2019

@author: cpetechstudent03
"""
# ///Convolutional Neural Network///


# Installing Keras
"""
pip install --upgrade keras
pip install --upgrade tensorflow
"""

### Part 1 - Building the CNN

# Importing Keras libraries and packages
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.utils.vis_utils import plot_model #vis
from keras.callbacks import ModelCheckpoint

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(filters = 32, kernel_size = (3, 3), input_shape=(64, 64, 3), activation="relu"))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - FLattener
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid')) #final layer

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

### Part 2 - Fitting to CNN to the images    

# //Model saving//
# Save the weights
"""
classifier.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
classifier.load_weights('./checkpoints/my_checkpoint')

# Save entire model to a HDF5 file
classifier.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
classifier = keras.models.load_model('my_model.h5')
classifier.summary() # read model
"""
#/checkpoints/
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)
# //end model saving//

# Image preprocessing   https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) #false if it matters

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64), # expected image size WILL VARY
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64), # expected image size WILL VARY
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=50,
        validation_data=test_set,
        validation_steps=2000, callbacks = [cp_callback])



#visualization
plot_model(classifier, to_file='model_plot.png', show_shapes=True, show_layer_names=True)