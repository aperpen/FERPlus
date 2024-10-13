import os
import sys
import math
import numpy as np
import tensorflow as tf

def build_model(num_classes, model_name):
    '''
    Factory function to instantiate the model.
    '''
    model = getattr(sys.modules[__name__], model_name)
    return model(num_classes)

class VGG13(object):
    '''
    A VGG13 like model (https://arxiv.org/pdf/1409.1556.pdf) tweaked for emotion data.
    '''
    @property
    def learning_rate(self):
        return 0.05

    @property
    def input_width(self):
        return 64

    @property
    def input_height(self):
        return 64

    @property
    def input_channels(self):
        return 1

    @property
    def model(self):
        return self._model

    def __init__(self, num_classes):
        self._model = self._create_model(num_classes)

    def _create_model(self, num_classes):
        model = tf.keras.Sequential([
            # Bloque 1
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(self.input_height, self.input_width, self.input_channels), name='conv1-1'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', name='conv1-2'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool1-1'),
            tf.keras.layers.Dropout(0.25, name='drop1-1'),

            # Bloque 2
            tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', name='conv2-1'),
            tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', name='conv2-2'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool2-1'),
            tf.keras.layers.Dropout(0.25, name='drop2-1'),

            # Bloque 3
            tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', name='conv3-1'),
            tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', name='conv3-2'),
            tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', name='conv3-3'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name='pool3-1'),
            tf.keras.layers.Dropout(0.25, name='drop3-1'),

            # Fully Connected Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation=None, name='fc5'),
            tf.keras.layers.ReLU(name='relu5'),
            tf.keras.layers.Dropout(0.5, name='drop5'),
            tf.keras.layers.Dense(1024, activation=None, name='fc6'),
            tf.keras.layers.ReLU(name='relu6'),
            tf.keras.layers.Dropout(0.5, name='drop6'),

            # Output Layer
            tf.keras.layers.Dense(num_classes, activation=None, name='output')
        ])
        return model
