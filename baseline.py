import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import optimizers
from keras import layers, Sequential
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from utils import get_custom_callback
import os
import sys


IMG_SIZE = 384  # this must correspond with what is in .h5 file
NUM_CLASSES = 5  # 5 output classes
NUM_EPOCHS = 50  # number of epochs
BATCH_SIZE = 5


def get_values():
    file = h5py.File('./data/data_rgb_384_processed.h5', 'r')
    return file['x_train'], file['y_train'], file['x_test'], file['y_test']


def main():
    # Name of this script
    script_name = os.path.basename(__file__)[0:-3]
    # Construct folder name using name of this script
    output_path_name = '_{}_outputs'.format(script_name)
    # Try to create a new folder
    try:
        # Make the output folder
        os.mkdir(output_path_name)
    except FileExistsError:
        pass

    # Model below this line ================================================

    custom_callback = get_custom_callback('multi_label', './{}'.format(output_path_name))
    callbacks_list = [custom_callback]


    x_train, y_train, x_test, y_test = get_values()

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=360
    )

    model = Sequential()

    densenet = DenseNet121(
        weights='./DenseNet-BC-121-32-no-top.h5',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  # optimizer=optimizers.Adam(lr=0.0001,decay=1e-6),
                  optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  metrics=['accuracy'])

    # fits the model on batches with real-time data augmentation:
    history = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, seed=1),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        max_queue_size=2
    )


if __name__ == '__main__':
    main()