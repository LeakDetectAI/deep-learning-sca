from numbers import Number

import tensorflow as tf


class LogTensorTF(object):
    def __init__(self, x):
        super(LogTensorTF, self).__init__()

        self._x = x
        self.add = _add_outofplace_tf
        self.imul = _imul_outofplace_tf

    def __add__(self, other):
        other_x = log_tf(other, like=self._x)
        return LogTensorTF(self.add(self._x, other_x))

    def __imul__(self, other):
        other_x = log_tf(other, like=self._x)
        self._x = self.imul(self._x, other_x)
        return self

    def __iadd__(self, other):
        other_x = log_tf(other, like=self._x)
        self._x = self.add(self._x, other_x)
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other_x = log_tf(other, like=self._x)
        diff = other_x - self._x
        x = self._x + log1mexp_tf(diff)
        return LogTensorTF(x)

    def __pow__(self, power):
        return LogTensorTF(self._x * power)

    def __mul__(self, other):
        other_x = log_tf(other, like=self._x)
        x = self._x + other_x
        return LogTensorTF(x)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        other_x = log_tf(other, like=self._x)
        x = self._x - other_x
        return LogTensorTF(x)

    def __truediv__(self, other):
        return self.__div__(other)

    def tf(self):
        return self._x

    def __repr__(self):
        tensor = self._x
        s = 'Log Tensor with value:\n{}'.format(tensor)
        return s


def log_tf(x, like):
    if isinstance(x, LogTensorTF):
        return x.tf()

    if not isinstance(x, Number):
        raise TypeError('Not supported type: received {}, '
                        'was expected LogTensorTF or Number'
                        .format(type(x)))

    data = like
    new = tf.broadcast_to(tf.math.log(tf.fill(1, x)), tf.shape(data))
    return new


def log1mexp_tf(U, eps=1e-3):
    res = tf.math.log1p(-tf.math.exp(U))
    small = tf.math.less(tf.math.abs(U), eps)
    res[small] = tf.math.log(-U[small])

    return res


def _imul_outofplace_tf(x1, x2):
    return x1 + x2


def _add_outofplace_tf(x1, x2):
    M = tf.math.maximum(x1, x2)
    return M + tf.math.log(tf.math.exp(x1 - M) + tf.math.exp(x2 - M))
