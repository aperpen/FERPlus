import sys
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes, model_name):
    '''
    Factory function to instantiate the model.
    '''
    model = getattr(sys.modules[__name__], model_name)
    return model(num_classes)


class VGG13:
    '''
    Un modelo similar a VGG13 (https://arxiv.org/pdf/1409.1556.pdf) ajustado para datos de emoción.
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
        model = models.Sequential()

        # Bloque 1 (Conv-Conv-Pool-Dropout)
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1-1'))
        model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1-2'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1-1'))
        model.add(layers.Dropout(0.25, name='drop1-1'))

        # Bloque 2 (Conv-Conv-Pool-Dropout)
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2-1'))
        model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2-2'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2-1'))
        model.add(layers.Dropout(0.25, name='drop2-1'))

        # Bloque 3 (Conv-Conv-Conv-Pool-Dropout)
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3-1'))
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3-2'))
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3-3'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool3-1'))
        model.add(layers.Dropout(0.25, name='drop3-1'))

        # Bloque 4 (Conv-Conv-Conv-Pool-Dropout)
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4-1'))
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4-2'))
        model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4-3'))
        model.add(layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool4-1'))
        model.add(layers.Dropout(0.25, name='drop4-1'))

        # Fully Connected Layers (Dense-Dropout)
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation=None, name='fc5'))
        model.add(layers.ReLU(name='relu5'))
        model.add(layers.Dropout(0.5, name='drop5'))

        model.add(layers.Dense(1024, activation=None, name='fc6'))
        model.add(layers.ReLU(name='relu6'))
        model.add(layers.Dropout(0.5, name='drop6'))

        # Output Layer
        model.add(layers.Dense(num_classes, activation=None, name='output'))

        return model
