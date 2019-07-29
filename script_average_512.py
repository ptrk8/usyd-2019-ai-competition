import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, Dense, concatenate, Flatten, BatchNormalization
from keras.models import Model
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import h5py
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from utils import get_custom_callback, to_multi_label
import os
import sys


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

    custom_callback = get_custom_callback('multi_label', './{}'.format(output_path_name))
    callbacks_list = [custom_callback]

    file = h5py.File('/albona/nobackup/andrewl/DeepLearning/data/data_rgb_512_processed.h5', 'r')
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

    densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    pool_1 = GlobalAveragePooling2D()(densenet.output)
    bat_1 = BatchNormalization(axis=1,momentum=0.1, epsilon=0.00001)(pool_1)
    drop_1 = Dropout(0.5)(bat_1)
    linear_1 = Dense(1024, activation = 'relu')(drop_1)
    bat_2 = BatchNormalization(axis=1,momentum=0.1, epsilon=0.00001)(linear_1)
    drop_2 = Dropout(0.5)(bat_2)
    final = Dense(NUM_CLASSES, activation = 'sigmoid')(drop_2)
    model = Model(inputs = densenet.input, outputs = final)
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
