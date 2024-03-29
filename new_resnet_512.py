import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras import optimizers
from keras import layers, Sequential
# https://github.com/keras-team/keras-contrib
from keras.applications.resnet50 import ResNet50
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py
from keras.callbacks import Callback, LearningRateScheduler
from sklearn.metrics import cohen_kappa_score
from utils import get_custom_callback, to_multi_label, best_lr_decay, f1_m, f1_loss, multi_label_acc
import sys
from keras.models import load_model

IMG_SIZE = 512  # this must correspond with what is in .h5 file
NUM_CLASSES = 5  # 5 output classes
NUM_EPOCHS = 50  # number of epochs
BATCH_SIZE = 8


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
    learn_rate = LearningRateScheduler(best_lr_decay, verbose=1)
    custom_callback = get_custom_callback('multi_label', './{}'.format(output_path_name))
    callbacks_list = [custom_callback, learn_rate]

    file = h5py.File('/albona/nobackup/andrewl/process/old_data_rgb_512_processed.h5', 'r')
    x_train, y_train, x_test, y_test = file['x_train'], file['y_train'], file['x_test'], file['y_test']

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    y_train = to_multi_label(y_train)
    y_test = to_multi_label(y_test)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=360
    )

    model = Sequential()

    resnet = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model.add(resnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  # optimizer=optimizers.Adam(lr=0.0001,decay=1e-6),
                  optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  metrics=[multi_label_acc, f1_m])

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
