import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
import pathlib

import sys
import json
import pandas as pd
import os
import argparse
import csv

from sklearn.metrics import confusion_matrix

from google.cloud import storage
from keras.datasets import fashion_mnist

from kubeflow.metadata import metadata
from datetime import datetime
from uuid import uuid4

# Helper libraries
import numpy as np


def set_arguments_from_list(arglist):
    parser = argparse.ArgumentParser()
    print('The argument recieved are {}'.format(arglist))
    learning_rate=0.01
    bucket_name='gs://'
    batch_size=64
    save_model=1
    epochs=1

    args = arglist.split('--')
    for arg in args: 
        if len(arg) ==0:
            continue
        subargs = arg.split()
        if len(subargs) == 2 and subargs[0] == 'learning_rate' :
            learning_rate = float(subargs[1])
        elif len(subargs) == 2 and subargs[0] == 'bucket_name' :
            bucket_name=subargs[1]
        elif len(subargs) == 2 and subargs[0] == 'epochs' :
            epochs=int(subargs[1])
        elif len(subargs) == 2 and subargs[0] == 'batch_size' :
            epochs=int(subargs[1])
    return (bucket_name,batch_size, epochs, save_model, learning_rate)

             

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name',
                        type=str,
                        default='gs://',
                        help='The bucket where the output has to be stored')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='Number of epochs for training the model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='the batch size for each epoch')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='the batch size for each epoch')
    parser.add_argument('--save_model',
                        type=int,
                        default=1,
                        help='the batch size for each epoch')

    args = parser.parse_known_args()[0]
    return args


def train(bucket_name, epochs=2, batch_size=512, learning_rate=0.001, save_model=1):

    testX, testy, trainX, trainy = load_and_normalize_data()
    cnn = create_tfmodel(optimizer=tf.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn.summary()

    cnn.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)

    predictions = cnn.predict(testX)

    pred = np.argmax(predictions, axis=1)

    test_loss, test_acc = cnn.evaluate(testX, testy, verbose=2)

    print("\naccuracy={}".format(test_acc))
    print("\nloss={} ".format(test_loss))

    if save_model == 1:
        save_tfmodel_in_gcs(exec, bucket_name, cnn, batch_size, epochs, test_loss, test_acc)


def save_tfmodel_in_gcs(exec, bucket_name, model, batch_size, epochs, test_loss, test_acc):

    export_path = bucket_name + '/export/model/1'
    tf.saved_model.save(model, export_dir=export_path)



def create_tfmodel(optimizer, loss, metrics):
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    cnn.add(tf.keras.layers.MaxPooling2D(2, 2))
    cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(64, activation='relu'))
    cnn.add(tf.keras.layers.Dense(10, activation='softmax'))
    cnn.compile(optimizer, loss, metrics)
    return cnn

  
def load_and_normalize_data():
    # load dataset
    (trainX, trainy), (testX, testy) = fashion_mnist.load_data()
    # Data Normalization - Dividing by 255 as the maximum possible value
    trainX = trainX / 255
    testX = testX / 255
    trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)
    testX = testX.reshape(testX.shape[0], 28, 28, 1)

    return testX, testy, trainX, trainy


if __name__ == '__main__':

    print("The arguments are ", str(sys.argv))
    if len(sys.argv) < 1:
        print("Usage: train bucket-name epochs batch-size")
        sys.exit(-1)

    if len(sys.argv) == 2:
        bucket_name,batch_size, epochs, save_model, learning_rate = set_arguments_from_list(sys.argv[1])
        print('About to train the model. bucket {} batch_size {} epochs {} save_model {} learning_rate {}'
                .format(bucket_name, batch_size, epochs, save_model, learning_rate))
        train(bucket_name, epochs, batch_size, learning_rate, save_model)
    else:
        args = parse_arguments()
        print(args)
        train(args.bucket_name, int(args.epochs), int(args.batch_size), float(args.learning_rate), int(args.save_model))
