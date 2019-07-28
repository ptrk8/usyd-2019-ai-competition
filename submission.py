import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from keras.applications.densenet import preprocess_input, DenseNet121
import numpy as np
import h5py
import pandas as pd
from keras.models import load_model
from utils import get_cur_milliseconds, get_best_preds, get_pessimist_preds


TEST_DATA_PATH_NAME = './data/data_rgb_512.h5'
MODEL_PATH_NAME = './models/1564258222240_kappa_0.8155_val_acc_0.938_acc_0.9531.h5'
SUBMISSION_PATH_NAME = './submissions'
BATCH_SIZE = 8


def create_submission(predictions, milliseconds, encoding_name):
    submission = pd.read_csv('./data/SampleSubmission.csv')
    submission['Expected'] = predictions
    submission.to_csv('{}/{}_{}_submission.csv'.format(SUBMISSION_PATH_NAME, milliseconds, encoding_name), index=False)


def get_sum_preds(predictions):
    return predictions.astype(int).sum(axis=1) - 1


def main():
    milliseconds = get_cur_milliseconds()
    # Retrieve values from file
    file = h5py.File(TEST_DATA_PATH_NAME, 'r')
    x_test = file['x_test']
    x_test = np.asarray(x_test)
    x_test = x_test.astype('float16')
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

    create_submission(y_pred_sum, milliseconds, 'sum_encoding')
    create_submission(y_pred_best, milliseconds, 'best_encoding')
    create_submission(y_pred_pessimist, milliseconds, 'pessimistic_encoding')


if __name__ == '__main__':
    main()