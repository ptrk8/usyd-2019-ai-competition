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
    multi_process, \
    f1_loss, \
    multi_label_acc, \
    f1_m, \
    get_ensemble_preds
import sys
from os import listdir
from os.path import isfile, join
from process import process_img, display_img
import cv2
from tqdm import tqdm
from halo import Halo
from pprint import pprint

TEST_DATA_PATH_NAME = './data/test_rgb_512_processed.h5'
SUBMISSION_FOLDER = './submissions'
ENSEMBLE_FOLDER = './ensemble'
BATCH_SIZE = 8


def create_submission(predictions, milliseconds, encoding_name, x_names):
    submission = pd.DataFrame()
    submission['Id'] = x_names
    submission['Expected'] = predictions
    submission.to_csv('{}/{}_{}_submission.csv'.format(SUBMISSION_FOLDER, milliseconds, encoding_name), index=False)


def model_predict(x_test, path_to_model, batch_size):
    model = load_model(path_to_model, custom_objects={'f1_loss': f1_loss,
                                                      'multi_label_acc': multi_label_acc,
                                                      'f1_m': f1_m})
    predictions = model.predict(x_test, batch_size=batch_size, verbose=1)
    # Returns boolean array of True and False [ True, False, False, ... ]
    return predictions > 0.5


# def process_img_batch(path_names):
#     return [process_img(path, cv2.IMREAD_COLOR, 384) for path in path_names]


def main():
    milliseconds = get_cur_milliseconds()
    # Retrieve values from file
    file = h5py.File(TEST_DATA_PATH_NAME, 'r')
    x_test = file['x_test']
    x_name = file['x_name']

    x_name = np.asarray(x_name).astype(str)
    x_test = np.asarray(x_test)
    x_test = x_test.astype('float16')

    model_names = listdir(ENSEMBLE_FOLDER)
    model_path_names = ['{}/{}'.format(ENSEMBLE_FOLDER, name) for name in model_names if isfile(join(ENSEMBLE_FOLDER, name))]
    # Returns array of arrays containing arrays with True and False [[[True, False...]], [...], ...]
    y_preds = [model_predict(x_test, path, BATCH_SIZE) for path in model_path_names]
    # Returns array of arrays containing values [[1, 4, 2 ...], [...], ... ]
    y_preds_sum = [get_sum_preds(pred) for pred in y_preds]
    y_preds_best = [get_best_preds(pred) for pred in y_preds]
    y_preds_pessimist = [get_pessimist_preds(pred) for pred in y_preds]

    y_pred_sum = get_ensemble_preds(y_preds_sum)
    y_pred_best = get_ensemble_preds(y_preds_best)
    y_pred_pessimist = get_ensemble_preds(y_preds_pessimist)

    create_submission(y_pred_sum, milliseconds, 'sum_encoding', x_name)
    create_submission(y_pred_best, milliseconds, 'best_encoding', x_name)
    create_submission(y_pred_pessimist, milliseconds, 'pessimistic_encoding', x_name)

if __name__ == '__main__':
    main()
