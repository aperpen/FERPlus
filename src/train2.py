import sys
import time
import os
import math
import csv
import argparse
import numpy as np
import logging

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from models2 import *
from ferplus import *

emotion_table = {'neutral'  : 0, 
                 'happiness': 1, 
                 'surprise' : 2, 
                 'sadness'  : 3, 
                 'anger'    : 4, 
                 'disgust'  : 5, 
                 'fear'     : 6, 
                 'contempt' : 7}
                 
# List of folders for training, validation and test.
train_folders = ['FER2013Train']
valid_folders = ['FER2013Valid']
test_folders = ['FER2013Test']


def cost_func(training_mode, prediction, target):
    if training_mode in ['majority', 'probability', 'crossentropy']:
        # Cross Entropy
        train_loss = tf.keras.losses.categorical_crossentropy(target, prediction)
    elif training_mode == 'multi_target':
        # Custom loss for multi-target
        train_loss = -tf.math.log(tf.reduce_max(target * prediction, axis=-1))
    return train_loss

@tf.function
def train_step(images, labels, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(labels, predictions))
    return loss, accuracy

def main(base_folder, training_mode='majority', model_name='VGG13', max_epochs=100):
    # create needed folders
    output_model_path = os.path.join(base_folder, 'models')
    output_model_folder = os.path.join(output_model_path, model_name + '_' + training_mode)
    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    # creating logging file
    logging.basicConfig(filename=os.path.join(output_model_folder, "train.log"), filemode='w', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info("Starting with training mode {} using {} model and max epochs {}.".format(
        training_mode, model_name, max_epochs))

    # create the model
    num_classes = len(emotion_table)
    model = build_model(num_classes, model_name)

    # set the input variables
    input_var = tf.keras.Input(shape=(model.input_height, model.input_width, 1))
    label_var = tf.keras.Input(shape=(num_classes,))

    # read FER+ dataset
    logging.info("Loading data...")
    train_params = FERPlusParameters(num_classes, model.input_height, model.input_width, training_mode, False)
    test_and_val_params = FERPlusParameters(num_classes, model.input_height, model.input_width, "majority", True)

    train_data_reader = FERPlusReader.create(base_folder, train_folders, "label.csv", train_params)
    val_data_reader = FERPlusReader.create(base_folder, valid_folders, "label.csv", test_and_val_params)
    test_data_reader = FERPlusReader.create(base_folder, test_folders, "label.csv", test_and_val_params)

    # print summary of the data
    display_summary(train_data_reader, val_data_reader, test_data_reader)

    # get the probabilistic output of the model
    pred = tf.keras.layers.Softmax()(model.model(input_var))

    epoch_size = train_data_reader.size()
    minibatch_size = 32

    # Training config
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[20*epoch_size, 40*epoch_size],
        values=[model.learning_rate, model.learning_rate / 2.0, model.learning_rate / 10.0]
    )
    optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)
    loss_fn = lambda y_true, y_pred: cost_func('multi_target', y_pred, y_true) # 'multi_target' -> training_mode
    # Loss and error
    model.model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    # Get minibatches of images to train with and perform model training
    max_val_accuracy = 0.0
    final_test_accuracy = 0.0
    best_test_accuracy = 0.0

    logging.info("Start training...")
    epoch = 0
    best_epoch = 0
    while epoch < max_epochs:
        train_data_reader.reset()
        val_data_reader.reset()
        test_data_reader.reset()

        # Training
        start_time = time.time()
        training_loss = 0
        training_accuracy = 0
        while train_data_reader.has_more():
            images, labels, current_batch_size = train_data_reader.next_minibatch(minibatch_size)

            # Train the model on the current batch
            train_step(images, labels, model.model, loss_fn, optimizer)

            # keep track of statistics
            metrics = model.model.evaluate(images, labels, verbose=0)
            training_loss += metrics[0] * current_batch_size
            training_accuracy += metrics[1] * current_batch_size

        training_accuracy /= train_data_reader.size()

        # Validation
        val_accuracy = 0
        while val_data_reader.has_more():
            images, labels, current_batch_size = val_data_reader.next_minibatch(minibatch_size)
            val_accuracy += model.model.evaluate(images, labels, verbose=0)[1] * current_batch_size

        val_accuracy /= val_data_reader.size()

        # if validation accuracy goes higher, we compute test accuracy
        test_run = False
        if val_accuracy > max_val_accuracy:
            best_epoch = epoch
            max_val_accuracy = val_accuracy
            model.model.save(os.path.join(output_model_folder, "model_{}".format(best_epoch)))

            test_run = True
            test_accuracy = 0
            while test_data_reader.has_more():
                images, labels, current_batch_size = test_data_reader.next_minibatch(minibatch_size)
                test_accuracy += model.model.evaluate(images, labels, verbose=0)[1] * current_batch_size

            test_accuracy /= test_data_reader.size()
            final_test_accuracy = test_accuracy
            if final_test_accuracy > best_test_accuracy:
                best_test_accuracy = final_test_accuracy

        logging.info("Epoch {}: took {:.3f}s".format(epoch, time.time() - start_time))
        logging.info("  training loss:\t{:e}".format(training_loss))
        logging.info("  training accuracy:\t\t{:.2f} %".format(training_accuracy * 100))
        logging.info("  validation accuracy:\t\t{:.2f} %".format(val_accuracy * 100))
        if test_run:
            logging.info("  test accuracy:\t\t{:.2f} %".format(test_accuracy * 100))

        epoch += 1

    logging.info("")
    logging.info("Best validation accuracy:\t\t{:.2f} %, epoch {}".format(max_val_accuracy * 100, best_epoch))
    logging.info("Test accuracy corresponding to best validation:\t\t{:.2f} %".format(final_test_accuracy * 100))
    logging.info("Best test accuracy:\t\t{:.2f} %".format(best_test_accuracy * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--base_folder",
                        type=str,
                        help="Base folder containing the training, validation and testing data.",
                        required=True)
    parser.add_argument("-m",
                        "--training_mode",
                        type=str,
                        default='majority',
                        help="Specify the training mode: majority, probability, crossentropy or multi_target.")

    args = parser.parse_args()
    main(args.base_folder, args.training_mode)
