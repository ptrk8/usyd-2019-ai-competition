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

IMG_SIZE = 384  # this must correspond with what is in .h5 file
NUM_CLASSES = 5  # 5 output classes
NUM_EPOCHS = 100  # number of epochs
BATCH_SIZE = 5

class Metrics(Callback):


    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1

        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        val_kappa = cohen_kappa_score(
            y_val,
            y_pred,
            weights='quadratic'
        )

        self.val_kappas.append(val_kappa)

        print("val_kappa: {}".format(val_kappa))

        # if _val_kappa == max(self.val_kappas):
        # print("Validation Kappa has improved. Saving model.")
        # self.model.save('model.h5')

        return


def get_values():
    file = h5py.File('./data/data_rgb_384_processed.h5', 'r')
    return file['x_train'], file['y_train'], file['x_test'], file['y_test']


def generate_imgs_from_file(path, batch_size):
    while True:
        file = h5py.File(path, 'r')
        xy_train = list(zip(file['x_train'], file['y_train']))
        len_xy_train = len(xy_train)
        batch_start = 0
        batch_end = batch_size
        while batch_start < len_xy_train:
            limit = min(batch_end, len_xy_train)
            x_train = [xy[0] for xy in xy_train[batch_start:limit]]
            y_train = [to_categorical(xy[1], NUM_CLASSES) for xy in xy_train[batch_start:limit]]
            yield(np.array(x_train), np.array(y_train))
            batch_start += batch_size
            batch_end += batch_size
        file.close()

        # for tpl in xy_train:
        #     yield(np.asarray([tpl[0]]), np.asarray([tpl[1]]))
        # file.close()


def main():


    x_train, y_train, x_test, y_test = get_values()

    # x_train = (n for n in x_train)
    # x_test = (n for n in x_test)
    # gc.collect()


    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)

    kappa_metrics = Metrics()
    kappa_metrics.val_kappas = []

    callbacks_list = [kappa_metrics]

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
        # generate_imgs_from_file('./data/data_rgb_384_processed.h5', BATCH_SIZE),
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, seed=1),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=(x_test, y_test),
        # validation_data=datagenVal.flow(x_test, y_test, batch_size=BATCH_SIZE),
        # validation_steps=len(x_test) // BATCH_SIZE,
        callbacks=callbacks_list,
        # use_multiprocessing=True,
        # workers=2
        max_queue_size=2
    )


if __name__ == '__main__':
    main()