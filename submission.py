import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from scipy import stats

from keras.applications.densenet import preprocess_input, DenseNet121
import numpy as np
import h5py
import pandas as pd
from keras.models import load_model
from utils import get_cur_milliseconds, \
    get_best_preds, \
    get_pessimist_preds, \
    get_sum_preds, \
    get_file_names_from_folder, \
    f1_loss, \
    multi_label_acc, \
    f1_m, \
    get_ensemble_preds
import sys

TEST_DATA_PATH_NAME = './data/data_rgb_512.h5'
SUBMISSION_FOLDER = './submissions'
ENSEMBLE_FOLDER = './ensemble'
BATCH_SIZE = 8


def create_submission(predictions, milliseconds, encoding_name):
    submission = pd.read_csv('./data/SampleSubmission.csv')
    submission['Expected'] = predictions
    submission.to_csv('{}/{}_{}_submission.csv'.format(SUBMISSION_FOLDER, milliseconds, encoding_name), index=False)


def model_predict(x_test, path_to_model, batch_size):
    model = load_model(path_to_model, custom_objects={'f1_loss': f1_loss,
                                                      'multi_label_acc': multi_label_acc,
                                                      'f1_m': f1_m})
    predictions = model.predict(x_test, batch_size=batch_size, verbose=1)
    # Returns boolean array of True and False [ True, False, False, ... ]
    return predictions > 0.5


def main():
    milliseconds = get_cur_milliseconds()
    # Retrieve values from file
    file = h5py.File(TEST_DATA_PATH_NAME, 'r')
    x_test = file['x_test']

    x_test = np.asarray(x_test).astype('float16')
    # Preprocess x_test data
    x_test = preprocess_input(x_test)

    file_names = get_file_names_from_folder(ENSEMBLE_FOLDER)
    if len(file_names) == 0 or len(file_names) % 2 == 0:
        print('Cant have no models and cant have even number of models')
        sys.exit(1)
    # Returns array of arrays containing arrays with True and False [[[True, False...]], [...], ...]
    y_preds = [model_predict(x_test, '{}/{}'.format(ENSEMBLE_FOLDER, file_name), BATCH_SIZE) for file_name in file_names]
    # Returns array of arrays containing values [[1, 4, 2 ...], [...], ... ]
    y_preds_sum = [get_sum_preds(pred) for pred in y_preds]
    y_preds_best = [get_best_preds(pred) for pred in y_preds]
    y_preds_pessimist = [get_pessimist_preds(pred) for pred in y_preds]

    y_pred_sum = get_ensemble_preds(y_preds_sum)
    y_pred_best = get_ensemble_preds(y_preds_best)
    y_pred_pessimist = get_ensemble_preds(y_preds_pessimist)

    create_submission(y_pred_sum, milliseconds, 'sum_encoding')
    create_submission(y_pred_best, milliseconds, 'best_encoding')
    create_submission(y_pred_pessimist, milliseconds, 'pessimistic_encoding')


if __name__ == '__main__':
    main()
