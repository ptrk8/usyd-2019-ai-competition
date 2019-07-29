import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.applications.densenet import preprocess_input, DenseNet121
import numpy as np
import h5py
import pandas as pd
from keras.models import load_model
from utils import get_cur_milliseconds, get_best_preds, get_pessimist_preds, get_sum_preds


TEST_DATA_PATH_NAME = './data/data_rgb_x_test_only_512.h5'
MODEL_PATH_NAME = './models/1564258222240_kappa_0.8155_val_acc_0.938_acc_0.9531.h5'
SUBMISSION_PATH_NAME = './submissions'
BATCH_SIZE = 8


def create_submission(predictions, milliseconds, encoding_name, names):
    submission = pd.DataFrame()
    submission['Id'] = names
    submission['Expected'] = predictions
    submission.to_csv('{}/{}_{}_submission.csv'.format(SUBMISSION_PATH_NAME, milliseconds, encoding_name), index=False)


def main():
    milliseconds = get_cur_milliseconds()
    # Retrieve values from file
    file = h5py.File(TEST_DATA_PATH_NAME, 'r')
    x_test = file['x_test']
    x_name = file['x_name']

    x_test = np.asarray(x_test)
    x_test = x_test.astype('float16')
    x_name = np.asarray(x_name).astype(str)
    # Preprocess x_test data
    x_test = preprocess_input(x_test)

    model = load_model(MODEL_PATH_NAME)
    predictions = model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
    # Returns boolean array of True and False [ True, False, False, ... ]
    y_pred = predictions > 0.5

    # This is the standard encoding where we sum each row
    y_pred_sum = get_sum_preds(y_pred)
    y_pred_best = get_best_preds(y_pred)
    y_pred_pessimist = get_pessimist_preds(y_pred)

    create_submission(y_pred_sum, milliseconds, 'sum_encoding', x_name)
    create_submission(y_pred_best, milliseconds, 'best_encoding', x_name)
    create_submission(y_pred_pessimist, milliseconds, 'pessimistic_encoding', x_name)


if __name__ == '__main__':
    main()