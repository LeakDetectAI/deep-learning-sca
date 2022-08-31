import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.metrics import binary_crossentropy

from .dice_bce_loss_helper import dice_loss
from .focal_loss_helper import sigmoid_focal_crossentropy
from .logarithm import LogTensorTF
from .topk_losses_helper import split_tf, log_sum_exp_k_autograd_tf, detect_large_tf

# Loss Function from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)
# Doesn't work with the current models due to library compatibility issues
# Made compatible by using y_pred scores rather than the layer itself
__all__ = ['ranking_loss', 'cross_entropy_ratio', 'binary_crossentropy_focal_loss',
           'categorical_crossentropy_focal_loss', 'bce_dice_loss', 'ranking_loss_optimized',
           'categorical_crossentropy_focal_loss_ratio', 'binary_crossentropy_focal_loss_ratio']
def ranking_loss(alpha_value=1.0, nb_class=256):
    def loss(y_true, y_pred):
        alpha = K.constant(alpha_value, dtype='float32')
        # Batch_size initialization
        y_true_int = K.cast(y_true, dtype='int32')
        batch_s = K.cast(K.shape(y_true_int)[0], dtype='int32')

        # Indexing the training set (range_value = (?,))
        range_value = K.arange(0, batch_s, dtype='int64')

        # Get rank and scores associated with the secret key (rank_sk = (?,))
        # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)
        values_topk_logits, indices_topk_logits = tf.nn.top_k(y_pred, k=nb_class, sorted=True)
        rank_sk = tf.where(tf.equal(K.cast(indices_topk_logits, dtype='int64'),
                                    tf.reshape(K.argmax(y_true_int), [tf.shape(K.argmax(y_true_int))[0], 1])))[:, 1] + 1
                                    # Index of the correct output among all the hypotheses (shape(?,))
        score_sk = tf.gather_nd(values_topk_logits, K.concatenate(
            [tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]),
             tf.reshape(rank_sk - 1, [tf.shape(rank_sk)[0], 1])]))  # Score of the secret key (shape(?,))

        # Ranking Loss Initialization
        loss_rank = 0

        for i in range(nb_class):
            # Score for each key hypothesis (s_i_shape=(?,))
            s_i = tf.gather_nd(values_topk_logits, K.concatenate(
                [tf.reshape(range_value, [tf.shape(values_topk_logits)[0], 1]),
                 i * tf.ones([tf.shape(values_topk_logits)[0], 1], dtype='int64')]))

            # Indicator function identifying when (i == secret key)
            indicator_function = tf.ones(batch_s) - (
                    K.cast(K.equal(rank_sk - 1, i), dtype='float32') * tf.ones(batch_s))

            # Logistic loss computation
            logistic_loss = K.log(1 + K.exp(-1 * alpha * (score_sk - s_i))) / K.log(2.0)

            # Ranking Loss computation
            loss_rank = tf.reduce_sum((indicator_function * logistic_loss)) + loss_rank

        return loss_rank / (K.cast(batch_s, dtype='float32'))

    return loss


def ranking_loss_optimized(alpha_value=1.0):
    def loss(y_true, y_pred):
        alpha = K.constant(alpha_value, dtype='float32')
        y_true = tf.cast(tf.math.argmax(y_true, axis=1), dtype=tf.int32)
        y_true_int = K.cast(y_true, dtype='int32')

        batch_s = K.cast(K.shape(y_true_int)[0], dtype='int32')
        batch_range = K.arange(0, batch_s, dtype='int32')

        # Get rank and scores associated with the secret key (rank_sk = (?,))
        # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)
        indices = tf.concat([batch_range[:, None], y_true_int[:, None]], 1)
        score_sk = tf.gather_nd(y_pred, indices)
        # Ranking Loss sk*-sk
        loss_rank_total = score_sk[:, None] - y_pred
        # Ranking Logistic Loss sum_{k \in K/k*} log2(1 + exp(sk*-sk))
        logistic_loss = K.log(1 + K.exp(-1 * alpha * (loss_rank_total))) / K.log(2.0)
        loss_rank = tf.reduce_sum(logistic_loss, axis=1) - - tf.constant(1.)
        return tf.reduce_mean(loss_rank)

    return loss


def ranking_loss_topk(alpha_value=10, k=256):
    def loss(y_true, y_pred):
        alpha = K.constant(alpha_value, dtype='float32')
        y_true = tf.cast(tf.math.argmax(y_true, axis=1), dtype=tf.int32)

        y_true_int = K.cast(y_true, dtype='int32')
        values_topk_logits, indices_topk_logits = tf.nn.top_k(y_pred, k=k, sorted=True)
        batch_s = K.cast(K.shape(y_true_int)[0], dtype='int32')
        batch_range = K.arange(0, batch_s, dtype='int32')

        # Get rank and scores associated with the secret key (rank_sk = (?,))
        # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)
        indices = tf.concat([batch_range[:, None], y_true_int[:, None]], 1)
        score_sk = tf.gather_nd(y_pred, indices)
        # Ranking Loss sk*-sk
        loss_rank_total = score_sk[:, None] - values_topk_logits
        # Ranking Logistic Loss sum_{k \in K/k*} log2(1 + exp(sk*-sk))
        logistic_loss = K.log(1 + K.exp(-1 * alpha * (loss_rank_total))) / K.log(2.0)
        loss_rank = tf.reduce_sum(logistic_loss, axis=1) - K.ones(batch_s, dtype='float32')
        return tf.reduce_mean(loss_rank)

    return loss


# Loss Function from Novel CER Paper as described in the Ranking Loss Paper
def cross_entropy_ratio():
    def loss(y_true, y_pred):
        # K.sum() for original_cce & shuffled_cce
        original_cce = K.sum(K.categorical_crossentropy(y_true, y_pred))
        shuffled_y_true = tf.random.shuffle(y_true)
        shuffled_cce = K.sum(K.categorical_crossentropy(shuffled_y_true, y_pred))
        cross_entropy_ratio_loss = original_cce / shuffled_cce
        return cross_entropy_ratio_loss

    return loss


def binary_crossentropy_focal_loss(alpha=0.25, gamma=2, from_logits=False):
    def loss(targets, inputs):
        sfce_obj = sigmoid_focal_crossentropy(y_true=targets, y_pred=inputs, alpha=alpha, gamma=gamma,
                                              from_logits=from_logits, base_loss='binary_crossentropy')
        return sfce_obj

    return loss


def categorical_crossentropy_focal_loss(alpha=0.25, gamma=2, from_logits=False):
    def loss(targets, inputs):
        sfce_obj = sigmoid_focal_crossentropy(y_true=targets, y_pred=inputs, alpha=alpha, gamma=gamma,
                                              from_logits=from_logits, base_loss='categorical_crossentropy')
        return sfce_obj

    return loss


def binary_crossentropy_focal_loss_ratio(alpha=0.25, gamma=2, from_logits=False):
    def loss(y_true, y_pred):
        # K.sum() for original_cce & shuffled_cce
        original_cce = binary_crossentropy_focal_loss(alpha=alpha, gamma=gamma, from_logits=from_logits)(y_true, y_pred)
        shuffled_y_true = tf.random.shuffle(y_true)
        shuffled_cce = binary_crossentropy_focal_loss(alpha=alpha, gamma=gamma, from_logits=from_logits)(
            shuffled_y_true, y_pred)
        cross_entropy_ratio_loss = original_cce / shuffled_cce
        return cross_entropy_ratio_loss

    return loss


def categorical_crossentropy_focal_loss_ratio(alpha=0.25, gamma=2, from_logits=False):
    def loss(y_true, y_pred):
        # K.sum() for original_cce & shuffled_cce
        original_cce = categorical_crossentropy_focal_loss(alpha=alpha, gamma=gamma, from_logits=from_logits)(y_true,
                                                                                                              y_pred)
        shuffled_y_true = tf.random.shuffle(y_true)
        shuffled_cce = categorical_crossentropy_focal_loss(alpha=alpha, gamma=gamma, from_logits=from_logits)(
            shuffled_y_true, y_pred)
        cross_entropy_ratio_loss = original_cce / shuffled_cce
        return cross_entropy_ratio_loss

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
