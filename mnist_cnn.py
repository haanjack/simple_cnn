'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import os

import tensorflow.compat.v1.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from nvtx.plugins.tf.keras.layers import NVTXStart, NVTXEnd
from nvtx.plugins.tf.keras.callbacks import NVTXCallback
import numpy as np

batch_size = 4
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28
# img_rows, img_cols = 224, 224

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    print('channels_first')
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    print('channels_last')
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = np.random.random(size=(60000, 256, 256, 1))
x_test = np.random.random(size=(10000, 256, 256, 1))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def get_model(input_shape=(28, 28, 1)):
    inputs = Input(input_shape)

    x = inputs
    # x = Reshape((1, 256, 256), input_shape=input_shape)(x)
    for kernelSize_y in [4, 8, 32, 64]:
        for kernelSize_x in [2, 7, 12, 17]:
            message='Conv2D (%d, %d)' % (kernelSize_x, kernelSize_y)
            print(message)
            x, marker_id, domain_id = NVTXStart(message=message, domain_name='forward', trainable=True)(x)
            x = Conv2D(filters=32, kernel_size=(kernelSize_x, kernelSize_y), activation='relu', padding='same')(x)
            x = NVTXEnd(grad_message=message, grad_domain_name='backwards')([x, marker_id, domain_id])

    x = Flatten()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model

nvtx_callback = NVTXCallback()
model = get_model((256, 256, 1))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[nvtx_callback],
          verbose=1,
          validation_data=(x_test, y_test))
