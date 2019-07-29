import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import optimizers
from keras import layers, Sequential
# from keras.applications.densenet import preprocess_input, DenseNet121
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from utils import get_custom_callback, to_multi_label, cohen_kappa_loss, f1_m, precision_m, recall_m
import os
import sys


IMG_SIZE = 256  # this must correspond with what is in .h5 file
NUM_CLASSES = 5  # 5 output classes
NUM_EPOCHS = 50  # number of epochs
BATCH_SIZE = 10


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

    file = h5py.File('./data/data_rgb_256_new.h5', 'r')
    x_train, y_train, x_test, y_test = file['x_train'], file['y_train'], file['x_test'], file['y_test']

    # x_train, x_test = preprocess_input(x_train), preprocess_input(x_test)
    x_train = x_train[0:500]
    y_train = y_train[0:500]

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    y_train = to_multi_label(y_train)
    y_test = to_multi_label(y_test)

    print(x_train.dtype)
    print(y_train.dtype)
    print(x_test.dtype)
    print(y_test.dtype)

    print('hello')

    datagen = ImageDataGenerator(
        # horizontal_flip=True,
        # vertical_flip=True,
        # rotation_range=360
    )

    model = Sequential()

    densenet = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    model.add(layers.Dense(5, activation='sigmoid'))

    model.summary()
    model_cohen_kappa = cohen_kappa_loss(num_classes=5)

    # model_cohen_kappa = get_cohen_kappa()
    model.compile(loss='binary_crossentropy',
                  # loss=model_cohen_kappa,
                  # optimizer=optimizers.Adam(lr=0.0001,decay=1e-6),
                  optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  metrics=['accuracy', f1_m])

    # return
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