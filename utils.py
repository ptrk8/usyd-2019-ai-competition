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
from multiprocessing import Pool
from scipy import stats


def get_ensemble_preds(predictions_lst):
    predictions_lst = np.asarray(predictions_lst)
    mode = stats.mode(predictions_lst, axis=0)
    mode_arr, count_arr = mode[0][0], mode[1][0]
    median_arr = np.median(predictions_lst, axis=0)
    # print('=============================================')
    # print(count_arr)
    # If there is complete disagreement across all, break ties by choosing median
    for idx, val in enumerate(count_arr):
        if val == 1:
            mode_arr[idx] = median_arr[idx]

    return mode_arr


def split_list(lst, n):
    """Splits list into equal chunks. Returns a generator. Use list() to convert to list. https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length"""
    k, m = divmod(len(lst), n)
    return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def flatten_list(lst):
    return [item for sublist in lst for item in sublist]


def multi_process(func, args, num_processes):
    """"First argument in args tuple must be the data."""
    if len(args) < 1:
        print('Parallelize has no args')
        sys.exit(1)
    chunks = list(split_list(args[0], num_processes))
    pool = Pool(processes=num_processes)
    # https://stackoverflow.com/questions/1993727/expanding-tuples-into-arguments
    starmap_args = map(lambda chunk: (chunk, *args[1:]), chunks)
    result = pool.starmap(func, starmap_args)
    pool.close()
    pool.join()
    return flatten_list(result)


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


def multi_label_acc(y_true, y_pred):
    y_true = K.cast(y_true, 'int32')
    y_pred = K.cast(y_pred + 0.5, 'int32')

    y_true = K.sum(y_true, axis=1) - 1
    y_pred = K.sum(y_pred, axis=1) - 1

    y_diff = K.cast(y_true - y_pred, 'int32')
    len_non_zero = K.cast(tf.math.count_nonzero(y_diff), 'int32')
    len_diff = tf.size(y_diff)

    return (len_diff - len_non_zero) / len_diff


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
    return tf.to_float(kappa)


def cohen_kappa_loss(num_classes=5, weights=None, metrics_collections=None, updates_collections=None, name=None):
    def cohen_kappa(y_true, y_pred):
        y_true = K.cast(y_true, 'int32')
        # Threshold our tensors
        y_pred = K.cast(y_pred + 0.5, 'int32')

        # y_true = tf.subtract(K.sum(y_true, axis=1), tf.constant(1))
        # y_pred = tf.subtract(K.sum(y_pred, axis=1), tf.constant(1))
        y_true = K.sum(y_true, axis=1) - 1
        y_pred = K.sum(y_pred, axis=1) - 1
        # y_pred = tf.cast(y_pred, tf.float32)
        # y_true = tf.cast(y_true, tf.float32)

        return 1 - _cohen_kappa(y_true, y_pred, num_classes, weights, metrics_collections, updates_collections, name)
    return cohen_kappa

# https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

# https://www.kaggle.com/christofhenkel/weighted-kappa-loss-for-keras-tensorflow
def kappa_loss_kaggle(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=10, name='kappa'):
    """A continuous differentiable approximation of discrete kappa loss.
        Args:
            y_pred: 2D tensor or array, [batch_size, num_classes]
            y_true: 2D tensor or array,[batch_size, num_classes]
            y_pow: int,  e.g. y_pow=2
            N: typically num_classes of the model
            bsize: batch_size of the training or validation ops
            eps: a float, prevents divide by zero
            name: Optional scope/name for op_scope.
        Returns:
            A tensor with the kappa loss."""
    # y_true = K.cast(y_true, 'int32')
    # # Threshold our tensors
    # y_pred = K.cast(y_pred + 0.5, 'int32')
    #
    # y_true = K.sum(y_true, axis=1)
    # y_pred = K.sum(y_pred, axis=1)
    with tf.name_scope(name):
        y_true = tf.to_float(y_true)
        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

        pred_ = y_pred ** y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(y_true, 0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /
                              tf.to_float(bsize))

        return -1 * nom / (denom + eps)


# https://github.com/openAGI/tefla
# https://openagi.github.io/tefla/core/losses/
def kappa_loss_tefla(predictions, labels, y_pow=1, eps=1e-15, num_ratings=5, batch_size=10, name='kappa'):
    with tf.name_scope(name):
        labels = tf.to_float(labels)
        repeat_op = tf.to_float(
            tf.tile(tf.reshape(tf.range(0, num_ratings), [num_ratings, 1]), [1, num_ratings]))
        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))
        weights = repeat_op_sq / tf.to_float((num_ratings - 1)**2)

        pred_ = predictions**y_pow
        try:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))
        except Exception:
            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [batch_size, 1]))

        hist_rater_a = tf.reduce_sum(pred_norm, 0)
        hist_rater_b = tf.reduce_sum(labels, 0)

        conf_mat = tf.matmul(tf.transpose(pred_norm), labels)

        nom = tf.reduce_sum(weights * conf_mat)
        denom = tf.reduce_sum(weights * tf.matmul(
            tf.reshape(hist_rater_a, [num_ratings, 1]), tf.reshape(hist_rater_b, [1, num_ratings]))
                / tf.to_float(batch_size))

        try:
            return -(1 - nom / denom)
        except Exception:
            return -(1 - nom / (denom + eps))


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
    try:
        val = float(val)
    except:
        pass

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
        arr = self.validation_data[:2]
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
        # print(logs_short)
        # Appends data to log file
        with open('{}/log.txt'.format(self.output_folder_path), 'a+') as log_file:
            log_file.write('{}\n'.format(json.dumps(logs_short)))

        self.model.save('{}/{}_kappa_{}_val_acc_{}_acc_{}.h5'.format(
            self.output_folder_path,
            curr_millisecond,
            logs_short['kappa'],
            logs_short['val_multi_label_acc'],
            logs_short['multi_label_acc']
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

