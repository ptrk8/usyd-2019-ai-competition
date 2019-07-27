from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from utils import get_custom_callback
import sys
import os

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

    image_height, image_width = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(60000, image_height * image_width)
    x_test = x_test.reshape(10000, image_height * image_width)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255.0
    x_test /= 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential()
    # Dense gives us fully connected nodes
    model.add(Dense(512, activation='relu', input_shape=(784,))) # the input layer
    model.add(Dense(512, activation='relu')) # keras knows there will be 512 input nodes
    # model.add(Dense(10, activation='softmax')) # outputs 10 nodes and each node will take on value between 0 and 1 which signifies confidence in that output
    model.add(Dense(10, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list
    )

if __name__ == '__main__':
    main()