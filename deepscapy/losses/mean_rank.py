import tensorflow as tf
from keras.metrics import Metric


class MeanRank(Metric):
    def __init__(self, name='mean_rank', **kwargs):
        super().__init__(name=name, **kwargs)
        self.predicted_ranks = None
        self.n_classes = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.math.argmax(y_true, axis=1), dtype=tf.int32)
        # Ranking
        n_classes = tf.shape(y_pred)[1]
        toprel, orderings = tf.math.top_k(y_pred, k=n_classes)
        troprel, rankings = tf.math.top_k(orderings, k=n_classes)
        rankings = tf.cast(rankings[:, ::-1], dtype=tf.int32)
        rankings = rankings + 1

        # Gather predicted ranks
        n_instances = tf.shape(y_true)[0]
        rankings_index = tf.expand_dims(tf.range(n_instances), axis=1)
        y_true_expanded = tf.expand_dims(y_true, axis=1)
        self.predicted_ranks = tf.cast(tf.gather_nd(rankings, tf.concat([rankings_index, y_true_expanded], axis=1)),
                                       dtype=tf.float32)
        self.n_classes = tf.cast(n_classes, dtype=tf.float32)

    def result(self):
        return tf.divide(tf.math.reduce_mean(self.predicted_ranks), self.n_classes)

    def reset_state(self):
        self.predicted_ranks = None

    def get_config(self):
        base_config = super().get_config()
        del base_config['name']
        return base_config

    def from_config(cls, config):
        return cls(**config)

class MeanRankMLC(Metric):
    def __init__(self, code_block, **kwargs):
        super().__init__(name='mean_rank_mlc', **kwargs)
        self.code_block = tf.cast(code_block, dtype=tf.float32)
        self.predicted_ranks = None
        self.n_classes = None

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_true = tf.reduce_sum((tf.expand_dims(y_true, 1) - tf.expand_dims(self.code_block, 0)) ** 2, 2)
        y_true = tf.cast(tf.math.argmin(y_true, axis=1), dtype=tf.int32)

        # Scores
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        y_pred = tf.reduce_sum((tf.expand_dims(y_pred, 1) - tf.expand_dims(self.code_block, 0)) ** 2, 2)

        # Ranking
        n_classes = tf.shape(self.code_block)[0]
        toprel, orderings = tf.math.top_k(y_pred, k=n_classes)
        troprel, rankings = tf.math.top_k(orderings, k=n_classes)
        rankings = tf.cast(rankings[:, ::-1], dtype=tf.int32)
        rankings = rankings + 1

        # Gather predicted ranks
        n_instances = tf.shape(y_true)[0]
        rankings_index = tf.expand_dims(tf.range(n_instances), axis=1)
        y_true_expanded = tf.expand_dims(y_true, axis=1)
        self.predicted_ranks = tf.cast(tf.gather_nd(rankings, tf.concat([rankings_index, y_true_expanded], axis=1)),
                                       dtype=tf.float32)
        self.n_classes = tf.cast(n_classes, dtype=tf.float32)

    def result(self):
        return tf.divide(tf.math.reduce_mean(self.predicted_ranks), self.n_classes)

    def reset_state(self):
        self.predicted_ranks = None

    def get_config(self):
        base_config = super().get_config()
        del base_config['name']
        return base_config

    def from_config(cls, config):
        return cls(**config)

class AccuracyMLC(Metric):
    def __init__(self, code_block, **kwargs):
        super().__init__(name='accuracy_mlc', **kwargs)
        self.is_correct = None
        self.code_block = tf.cast(code_block, dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true,  dtype=tf.float32)
        y_true = tf.reduce_sum((tf.expand_dims(y_true, 1) - tf.expand_dims(self.code_block, 0)) ** 2, 2)
        y_true = tf.cast(tf.math.argmin(y_true, axis=1), dtype=tf.int32)

        # Scores
        y_pred = tf.cast(y_pred,  dtype=tf.float32)
        y_pred = tf.reduce_sum((tf.expand_dims(y_pred, 1) - tf.expand_dims(self.code_block, 0)) ** 2, 2)
        y_pred = tf.cast(tf.math.argmin(y_pred, axis=1), dtype=tf.int32)
        self.is_correct = tf.cast(tf.math.equal(y_pred, y_true), tf.float32)

    def result(self):
        return tf.reduce_mean(self.is_correct)

    def reset_state(self):
        self.is_correct = None

    def get_config(self):
        base_config = super().get_config()
        del base_config['name']
        return base_config

    def from_config(cls, config):
        return cls(**config)
