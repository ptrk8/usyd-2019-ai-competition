from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
import numpy as np
import sys
import json
import pytz
from datetime import datetime
import time
import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import keras.backend as K

def multi(arr):
    arr_new = np.copy(arr)

    for idx, val in enumerate(arr_new):
        if val == 1:
            break
        arr_new[idx] = 1

    return arr_new.astype('int16').tolist()


def to_multi_label(arr):
    return np.asarray([multi(output) for output in arr])


def get_multi_label_outputs(arr):
    return np.copy(arr).astype(int).sum(axis=1) - 1


def get_multi_class_outputs(arr):
    return np.argmax(arr, axis=1)


def get_curr_datetime():
    tz = pytz.timezone('Australia/Melbourne')
    now = datetime.now(tz)
    return now.strftime("%m/%d/%Y, %H:%M:%S")


def get_file_names_from_folder(folder_name):
    return [f for f in listdir(folder_name) if isfile(join(folder_name, f))]


def get_cur_milliseconds():
    return int(round(time.time() * 1000))


def get_best_preds(arr):
    arr = np.asarray(arr).astype(int)
    mask = arr == 0
    return np.clip(np.where(mask.any(1), mask.argmax(1), 5) - 1, 0, 4)


def get_score(prediction):
    for idx, val in reversed(list(enumerate(prediction))):
        if val:
            return idx
    return 0


def get_pessimist_preds(predictions):
    return [get_score(prediction) for prediction in predictions]


def get_sum_preds(predictions):
    return predictions.astype(int).sum(axis=1) - 1


# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# # https://stackoverflow.com/questions/54831044/how-can-i-specify-a-loss-function-to-be-quadratic-weighted-kappa-in-keras
# def get_cohen_kappa(weights='quadratic'):
#     def _cohen_kappa_score(y_val, y_pred):
#         print(y_val)
#         # Get y_val values as array of integers [ 1, 2, 5 ]
#         y_val = np.asarray(y_val).sum(axis=1) - 1
#         y_pred_bool = y_pred > 0.5
#         # Get y_pred values as array of boolean values [ True, False etc. ]
#         y_pred_sum = get_sum_preds(y_pred_bool)
#         score = cohen_kappa_score(y_val, y_pred_sum, weights=weights)
#
#         # score = cohen_kappa_score(y_val, y_pred, weights='quadratic')
#         return -1 * score
#     return _cohen_kappa_score

# def _cohen_kappa(y_true, y_pred, num_classes=5, weights=None, metrics_collections=None, updates_collections=None, name=None):
#     return tf.convert_to_tensor(cohen_kappa_score(K.eval(y_true), K.eval(y_pred), weights='quadratic'))

def _cohen_kappa(y_true, y_pred, num_classes=5, weights=None, metrics_collections=None, updates_collections=None, name=None):
    kappa, update_op = tf.contrib.metrics.cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)
    # kappa = K.cast(kappa, 'float32')
    K.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([update_op]):
        kappa = tf.identity(kappa)
    return kappa


def cohen_kappa_loss(num_classes=5, weights=None, metrics_collections=None, updates_collections=None, name=None):
    def cohen_kappa(y_true, y_pred):
        y_true = K.cast(y_true, 'int32')
        # Threshold our tensors
        y_pred = K.cast(y_pred + 0.5, 'int32')

        y_true = K.sum(y_true, axis=1)
        y_pred = K.sum(y_pred, axis=1)
        # y_pred = tf.cast(y_pred, tf.float32)
        # y_true = tf.cast(y_true, tf.float32)

        return -_cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)
    return cohen_kappa


# https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
# https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def best_lr_decay(epoch):
    """Use with adam optimizer to mimic bestfitting"""
    lr = 0.0003
    if epoch > 25:
        lr = 0.00015
    if epoch > 30:
        lr = 0.000075
    if epoch > 35:
        lr = 0.00003
    if epoch > 40:
        lr = 0.00001
    return lr


def _round(val, decimals):
    if not isinstance(val, str):
        return round(val, decimals)
    return val

class Metrics(Callback):

    def on_epoch_end(self, epoch, logs={}):
        # Get Current time
        curr_datetime = get_curr_datetime()
        curr_millisecond = get_cur_milliseconds()
        # Add current time to logs
        logs['datetime'] = curr_datetime
        logs['millisecond'] = curr_millisecond
        # Get x_validation and y_validation
        x_val, y_val = self.validation_data[:2]
        if self.out_type == 'multi_label':
            y_val = y_val.sum(axis=1) - 1
            y_pred = self.model.predict(x_val) > 0.5
            y_pred_sum = get_sum_preds(y_pred)
            # Get quadratic weighted kappa
            logs['kappa'] = cohen_kappa_score(y_val, y_pred_sum, weights='quadratic')
            # Get kappa for bestfittings method of encoding
            y_pred_bestfitting = get_best_preds(y_pred)
            logs['kappa_bestfitting'] = cohen_kappa_score(y_val, y_pred_bestfitting, weights='quadratic')
            print("val_kappa_bestfitting: {}".format(logs['kappa_bestfitting']))
            # Get kappa for bestfittings method of encoding
            y_pred_pessimistic = get_pessimist_preds(y_pred)
            logs['kappa_pessimistic'] = cohen_kappa_score(y_val, y_pred_pessimistic, weights='quadratic')
            print("val_kappa_pessimistic: {}".format(logs['kappa_pessimistic']))
        elif self.out_type == 'multi_class':
            y_val = get_multi_class_outputs(y_val)
            y_pred = self.model.predict(x_val)
            y_pred = get_multi_class_outputs(y_pred)
            print(y_val.shape)
            print(y_pred.shape)
            # Get quadratic weighted kappa
            logs['kappa'] = cohen_kappa_score(y_val, y_pred, weights='quadratic')

        # Add kappa to logs
        # logs['kappa'] = val_kappa
        # self.val_kappas.append(val_kappa)
        logs_short = {key: _round(val, 4) for key, val in logs.items()}
        # Appends data to log file
        with open('{}/log.txt'.format(self.output_folder_path), 'a+') as log_file:
            log_file.write('{}\n'.format(json.dumps(logs_short)))

        self.model.save('{}/{}_kappa_{}_val_acc_{}_acc_{}.h5'.format(
            self.output_folder_path,
            curr_millisecond,
            logs_short['kappa'],
            logs_short['val_acc'],
            logs_short['acc']
        ))

        # if _val_kappa == max(self.val_kappas):
        # print("Validation Kappa has improved. Saving model.")
        # self.model.save('model.h5')
        # Print Kappa
        print("val_kappa: {}".format(logs['kappa']))
        return


def get_custom_callback(out_type='multi_label', output_folder_path=None):
    if not output_folder_path:
        print('No output folder path was provided')
        sys.exit(1)
    callback = Metrics()
    # callback.val_kappas = []
    callback.out_type = out_type
    callback.output_folder_path = output_folder_path
    return callback

