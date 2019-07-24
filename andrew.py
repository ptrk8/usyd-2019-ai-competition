

from keras import optimizers
from keras import layers, Sequential
from keras.applications.densenet import preprocess_input, DenseNet121
from keras.utils.np_utils import to_categorical

import numpy as np
import h5py


def get_values():
    file = h5py.File('/tmp/{}'.format('data_rgb_384_new_processed.h5'), 'r')
    x_train, y_train, x_test, y_test = file['x_train'], file['y_train'], file['x_test'], file['y_test']

def main():




    IMG_SIZE = 384  # this must correspond with what is in .h5 file
    NUM_CLASSES = 5  # 5 output classes
    NUM_EPOCHS = 100  # number of epochs
    BATCH_SIZE = 32

    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    model = Sequential()
    model.add(densenet)

    densenet = DenseNet121(
        weights='./DenseNet-BC-121-32-no-top',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  #               optimizer=optimizers.Adam(lr=0.0001,decay=1e-6),
                  optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
                  metrics=['accuracy'])

    # fits the model on batches with real-time data augmentation:
    history = model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=12),
        steps_per_epoch=len(x_train) / 12,
        epochs=NUM_EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        max_queue_size=2
    )


if __name__ == '__main__':
    main()