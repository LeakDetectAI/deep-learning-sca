import numpy as np
import tensorflow as tf
from keras.losses import binary_crossentropy

from deepscapy.losses.dice_bce_loss_helper import dice_loss
from deepscapy.losses.focal_loss_helper import sigmoid_focal_crossentropy
from deepscapy.losses.logarithm import LogTensorTF
from deepscapy.losses.topk_losses_helper import split_tf, log_sum_exp_k_autograd_tf, detect_large_tf


def binary_crossentropy_focal_loss(alpha=0.25, gamma=2, from_logits=False):
    def loss(targets, inputs):
        sfce_obj = sigmoid_focal_crossentropy(y_true=targets, y_pred=inputs, alpha=alpha, gamma=gamma,
                                              from_logits=from_logits, base_loss='binary_crossentropy')
        return sfce_obj

    return loss


def binary_crossentropy_focal_loss_ratio(alpha=0.25, gamma=2, from_logits=False, n=1):
    def loss(y_true, y_pred):
        flbce_orignal = binary_crossentropy_focal_loss(alpha=alpha, gamma=gamma, from_logits=from_logits)(y_true,
                                                                                                          y_pred)
        flbce_shuffled = 0.0
        for i in range(n):
            shuffled_y_true = tf.random.shuffle(y_true)
            flbce_shuffled += binary_crossentropy_focal_loss(alpha=alpha, gamma=gamma, from_logits=from_logits)(
                shuffled_y_true, y_pred)
        flbce_shuffled = flbce_shuffled / n
        flbce_ratio_loss = flbce_orignal / flbce_shuffled
        return flbce_ratio_loss

    return loss

# Source: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
def bce_dice_loss(smooth=1.0):
    def loss(y_true, y_pred):
        dice_bce_loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred, smooth)
        return dice_bce_loss / 2.0

    return loss

# Loss function made compatible with Tensorflow (Pytorch Loss Source: https://github.com/oval-group/smooth-topk)
def Topk_Smooth_SVM(num_classes=256, k=1, tau=1., alpha=1.):
    def loss(y_true, y_pred):
        labels = tf.cast(tf.range(num_classes), dtype=tf.int32)
        y_true = tf.cast(tf.math.argmax(y_true, axis=1), dtype=tf.int32)

        x_1, x_2 = split_tf(y_pred, y_true, labels)
        x_1 = tf.cast(tf.math.divide(x_1, k * tau), dtype=tf.float64)
        x_2 = tf.cast(tf.math.divide(x_2, k * tau), dtype=tf.float64)

        res = log_sum_exp_k_autograd_tf(x_1, k)
        term_1, term_2 = res[1], res[0]
        term_1, term_2 = LogTensorTF(term_1), LogTensorTF(term_2)

        X_2 = LogTensorTF(x_2)
        cst = tf.cast(tf.fill([1], float(alpha) / tau), tf.float64)
        One_by_tau = LogTensorTF(cst)
        Loss_ = term_2 * X_2

        loss_pos = (term_1 * One_by_tau + Loss_).tf()
        loss_neg = Loss_.tf()
        loss_smooth = tau * (loss_pos - loss_neg)

        return loss_smooth

    return loss

# Loss function made compatible with Tensorflow (Pytorch Loss Source: https://github.com/oval-group/smooth-topk)
def Topk_Hard_SVM(num_classes=256, k=1, alpha=1.):
    def loss(y_true, y_pred):
        labels = tf.cast(tf.range(num_classes), dtype=tf.int32)
        y_true = tf.cast(tf.math.argmax(y_true, axis=1), dtype=tf.int32)

        x_1, x_2 = split_tf(y_pred, y_true, labels)

        max_1, _ = tf.math.top_k(x_1 + alpha, k=k)
        max_1 = tf.math.reduce_mean(max_1, axis=1)

        max_2 = tf.math.top_k(x_1, k=k - 1).values
        max_2 = (tf.math.reduce_sum(max_2, axis=1) + x_2) / k

        loss_hard = tf.clip_by_value(max_1 - max_2, 0, np.inf)

        return loss_hard

    return loss

# Loss function made compatible with Tensorflow (Pytorch Loss Source: https://github.com/oval-group/smooth-topk)
def SmoothTopkSVM(n_classes=256, alpha=1, tau=1., k=5, thresh=1e3):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.math.argmax(y_true, axis=1), dtype=tf.int32)
        smooth, hard = detect_large_tf(y_pred, k, tau, thresh)
        total_loss = 0

        smooth_int = tf.cast(smooth, dtype=tf.int32)
        hard_int = tf.cast(hard, dtype=tf.int32)

        if tf.cast(tf.math.reduce_sum(smooth_int), dtype=tf.bool):
            x_s, y_s = y_pred[smooth], y_true[smooth]
            x_s = tf.reshape(x_s, [-1, tf.shape(y_pred)[1]])
            # y_s = to_categorical(y_s, num_classes=n_classes)
            y_s = tf.one_hot(y_s, n_classes)
            total_loss += tf.math.reduce_sum(
                Topk_Smooth_SVM(num_classes=n_classes, k=k, tau=tau, alpha=alpha)(y_s, x_s)) / tf.cast(
                tf.shape(y_pred)[0], dtype=tf.float64)

        if tf.cast(tf.math.reduce_sum(hard_int), dtype=tf.bool):
            x_h, y_h = y_pred[hard], y_true[hard]
            x_h = tf.reshape(x_h, [-1, tf.shape(y_pred)[1]])
            # y_h = to_categorical(y_h, num_classes=n_classes)
            y_h = tf.one_hot(y_h, n_classes)
            total_loss += tf.math.reduce_sum(
                Topk_Hard_SVM(num_classes=n_classes, k=k, alpha=alpha)(y_h, x_h)) / tf.cast(tf.shape(y_pred)[0],
                                                                                            dtype=tf.float64)

        return total_loss

    return loss
