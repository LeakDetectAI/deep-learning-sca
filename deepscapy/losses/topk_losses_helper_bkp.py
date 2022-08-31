import itertools
import math
import operator
from functools import reduce

import tensorflow as tf
from future.builtins import range

from .logarithm import LogTensorTF


def split_tf(x, y, labels):
    mask = tf.math.not_equal(labels[None, :], y[:, None])

    x_1 = tf.reshape(x[mask], [tf.shape(x)[0], -1])
    x_2 = tf.reshape(tf.gather(x, y[:, None], batch_dims=1), [-1])

    return x_1, x_2


def divide_and_conquer_tf(x, k, mul):
    # to_merge = []
    to_merge = tf.TensorArray(tf.float64, size=0, dynamic_size=True, clear_after_read=False)

    x_dim = tf.rank(x[0])
    x_size = tf.shape(x[0])[0]

    while x_dim > 1 and x_size > 1:
        size = tf.shape(x[0])[0]
        half = size // 2
        if 2 * half < size:
            # to_merge.append([t[-1] for t in x])
            # [t[-1] for t in x]
            to_merge = to_merge.write(to_merge.size(), tf.map_fn(lambda t: t[-1], x))
        # [t[:half] for t in x], [t[half: 2 * half] for t in x]
        x = mul(tf.map_fn(lambda t: t[:half], x),
                tf.map_fn(lambda t: t[half: 2 * half], x))
        x_dim = tf.rank(x[0])
        x_size = tf.shape(x[0])[0]

    # for row in to_merge:
    #     x = mul(x, row)
    for index in range(to_merge.size()):
        x = mul(x, to_merge.read(index))
    x = tf.concat(x, 0)
    return x


def Multiplication_tf(k):
    assert isinstance(k, int) and k > 0

    def isum(factors):
        init = next(factors)
        return reduce(operator.iadd, factors, init)

    def mul_function(x1, x2):
        # prepare indices for convolution
        l1, l2 = len(x1), len(x2)
        M = min(k + 1, l1 + l2 - 1)
        indices = [[] for _ in range(M)]
        for (i, j) in itertools.product(range(l1), range(l2)):
            if i + j >= M:
                continue
            indices[i + j].append((i, j))

        # wrap with log-tensors for stability
        X1 = [LogTensorTF(x1[i]) for i in range(l1)]
        X2 = [LogTensorTF(x2[i]) for i in range(l2)]

        # perform convolution
        coeff = tf.TensorArray(tf.float64, size=0, dynamic_size=True, clear_after_read=False)
        for c in range(M):
            coeff = coeff.write(coeff.size(), isum(X1[i] * X2[j] for (i, j) in indices[c]).tf())
            # coeff[c].assign(isum(X1[i] * X2[j] for (i, j) in indices[c]).tf())
            # coeff.append(isum(X1[i] * X2[j] for (i, j) in indices[c]).tf())
        coeff = tf.Variable(coeff.stack())
        return coeff

    return mul_function


def log_sum_exp_k_autograd_tf(x, k):
    n_s = tf.shape(x)[0]

    # assert k <= tf.shape(x)[1]

    x_summed = tf.cast(tf.math.reduce_sum(x, axis=1), dtype=tf.float64)
    x_trans = tf.transpose(x)
    x_inv_log = tf.cast(tf.math.scalar_mul(-1, x_trans), tf.float64)
    x_combined = tf.Variable([x_inv_log, tf.cast(tf.fill(tf.shape(x_inv_log), 0.0), tf.float64)])

    log_res = divide_and_conquer_tf(x_combined, k, mul=Multiplication_tf(k))

    coeff = log_res + x_summed[None, :]
    coeff = tf.reshape(coeff, [k + 1, n_s])

    return coeff[k - 1: k + 1]


def detect_large_tf(x, k, tau, thresh):
    top, _ = tf.math.top_k(x, k=k + 1)

    hard = tf.stop_gradient(tf.math.greater_equal(top[:, k - 1] - top[:, k], k * tau * math.log(thresh)))
    smooth = tf.math.equal(hard, False)

    return smooth, hard
