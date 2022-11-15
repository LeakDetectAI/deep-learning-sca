import tensorflow as tf
from keras import backend as K
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from tensorflow_addons.losses import sigmoid_focal_crossentropy
@tf.function
def sigmoid_focal_crossentropy(
        y_true: TensorLike,
        y_pred: TensorLike,
        alpha: FloatTensorLike = 0.25,
        gamma: FloatTensorLike = 2.0,
        from_logits: bool = False,
        base_loss='categorical_crossentropy'
) -> tf.Tensor:
    """Implements the focal loss function.

    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much higher for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.

    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.

    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    if base_loss == 'categorical_crossentropy':
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Cross-entropy
        cross_entropy = -tf.reduce_sum(y_true * K.log(y_pred + epsilon))
    else:
        cross_entropy = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    if base_loss == 'categorical_crossentropy':
        p_t = (y_true * pred_prob)
    else:
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))

    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * cross_entropy, axis=-1)
