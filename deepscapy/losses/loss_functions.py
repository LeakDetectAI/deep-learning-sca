import tensorflow as tf
from keras import backend as K

# Loss Function from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)
# Doesn't work with the current models due to library compatibility issues
# Made compatible by using y_pred scores rather than the layer itself
__all__ = ['ranking_loss', 'cross_entropy_ratio', 'categorical_crossentropy_focal_loss', 'ranking_loss_optimized',
           'categorical_crossentropy_focal_loss_ratio']

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

        # Indexing the training set (range_value = (?,))
        batch_range = K.arange(0, batch_s, dtype='int32')

        # Get rank and scores associated with the secret key (rank_sk = (?,))
        # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)
        indices = tf.concat([batch_range[:, None], y_true[:, None]], 1)
        score_sk = tf.gather_nd(y_pred, indices)
        # Ranking Loss sk*-sk
        loss_rank_total = score_sk[:, None] - y_pred
        # Ranking Logistic Loss sum_{k \in K/k*} log2(1 + exp(sk*-sk))
        logistic_loss = K.log(1 + K.exp(-1 * alpha * (loss_rank_total))) / K.log(2.0)
        loss_rank = tf.reduce_sum(logistic_loss, axis=1) - 1.0  # K.ones(batch_s, dtype='float32')
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
def cross_entropy_ratio(n=1):
    def loss(y_true, y_pred):
        # K.sum() for orignal_cee & shuffled_cce
        orignal_cce = K.categorical_crossentropy(y_true, y_pred)
        ce_shuffled = 0.0
        for i in range(n):
            y_true_shuffled = tf.random.shuffle(y_true)
            ce_shuffled += tf.keras.losses.categorical_crossentropy(y_true_shuffled, y_pred)
        ce_shuffled = ce_shuffled / n
        cross_entropy_ratio_loss = orignal_cce / ce_shuffled
        return cross_entropy_ratio_loss

    return loss


def categorical_crossentropy_focal_loss(alpha=0.25, gamma=2, from_logits=False):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        if from_logits:
            y_pred = tf.sigmoid(y_pred)

        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return loss


def categorical_crossentropy_focal_loss_ratio(alpha=0.25, gamma=2, from_logits=False, n=1):
    def loss(y_true, y_pred):
        flcce_orignal = categorical_crossentropy_focal_loss(alpha=alpha, gamma=gamma, from_logits=from_logits)(y_true,
                                                                                                               y_pred)
        flcce_shuffled = 0.0
        for i in range(n):
            shuffled_y_true = tf.random.shuffle(y_true)
            flcce_shuffled += categorical_crossentropy_focal_loss(alpha=alpha, gamma=gamma, from_logits=from_logits)(
                shuffled_y_true, y_pred)
        flcce_shuffled = flcce_shuffled / n
        flcce_ratio_loss = flcce_orignal / flcce_shuffled
        return flcce_ratio_loss
    return loss
