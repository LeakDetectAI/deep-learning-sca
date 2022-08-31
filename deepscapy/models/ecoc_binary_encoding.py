import logging

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import RobustScaler

from sklearn.utils import check_random_state
from keras.layers import Flatten, Dense, Input, Activation, AveragePooling1D, Conv1D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow_addons.metrics import HammingLoss

from deepscapy.core.sca_nn_model import SCANNModel, ONED_CNN
from deepscapy.losses.loss_functions import binary_crossentropy_focal_loss
from deepscapy.losses.mean_rank import AccuracyMLC, MeanRankMLC


# Model Code from Ranking Loss Paper Github Repository (https://github.com/gabzai/Ranking-Loss-SCA)

class CNNECOCNNModel(SCANNModel):
    def __init__(self, model_name, num_classes, features, code_size, random_state=42,
                 loss_function=binary_crossentropy_focal_loss(), kernel_regularizer=l2(l=1e-8),
                 kernel_initializer="he_uniform", optimizer=Adam(learning_rate=1e-2), metrics=None,
                 weight_averaging=False, **kwargs):

        self.code_size = code_size
        self.code_length = int(num_classes * self.code_size)
        self.random_state = check_random_state(random_state)
        remake = True
        while remake:
            self.ecoc_code_block = self.random_state.uniform(size=(num_classes, self.code_length))
            self.ecoc_code_block[self.ecoc_code_block > 0.5] = 1
            self.ecoc_code_block[self.ecoc_code_block != 1] = 0
            self.ecoc_code_block = np.array(self.ecoc_code_block, dtype=np.int)
            if np.unique(self.ecoc_code_block, axis=0).shape[0] == num_classes:
                remake = False

        self.classes_index = {c: i for i, c in enumerate(np.arange(num_classes))}
        self.features = features
        #input_dim = self.get_reshaped_input_dim()
        if metrics is None:
            metrics = [AccuracyMLC(code_block=self.ecoc_code_block), MeanRankMLC(code_block=self.ecoc_code_block),
                       HammingLoss(mode='multilabel', threshold=0.5)]
        super(CNNECOCNNModel, self).__init__(model_name=model_name, num_classes=num_classes, input_dim=features,
                                             model_type=ONED_CNN, loss_function=loss_function,
                                             kernel_regularizer=kernel_regularizer,
                                             kernel_initializer=kernel_initializer, optimizer=optimizer,
                                             metrics=metrics, weight_averaging=weight_averaging, **kwargs)
        self.logger = logging.getLogger(CNNECOCNNModel.__name__)
        self.logger.info("ECOC Code Block: \n {}".format(self.ecoc_code_block))
        self.logger.info("Relative code size {:.4f} code_length {}".format(self.code_size, self.code_length))

    def get_reshaped_input_dim(self):
        image_size = int(np.floor(np.sqrt(self.features)) + 1)
        input_shape = (image_size, image_size, 1)
        return input_shape

    def reshape_inputs(self, X, y):
        X, _ = super(CNNECOCNNModel, self).reshape_inputs(X, None)
        if y is not None:
            y = np.array([self.ecoc_code_block[self.classes_index[y[i]]] for i in range(y.shape[0])], dtype=int)
        return X, y

    def _construct_model_(self, **kwargs):
        input_shape = (self.input_dim, 1)
        img_input = Input(shape=input_shape)

        # 1st convolutional block
        x = AveragePooling1D(2, strides=2)(img_input)
        x = Conv1D(32, 1, activation='selu', padding='same', name='block1_conv1_ascad_desync50', **kwargs)(
            img_input)
        x = BatchNormalization(name='block1_batchnorm_ascad_desync50')(x)
        x = AveragePooling1D(2, strides=2, name='block1_pool_ascad_desync50')(x)

        # 2nd convolutional block
        x = Conv1D(64, 25, activation='selu', padding='same', name='block2_conv1_ascad_desync50', **kwargs)(x)
        x = BatchNormalization(name='block2_batchnorm_ascad_desync50')(x)
        x = AveragePooling1D(25, strides=25, name='block2_pool_ascad_desync50')(x)

        # 3rd convolutional block
        x = Conv1D(128, 3, activation='selu', padding='same', name='block3_conv1_ascad_desync50', **kwargs)(x)
        x = BatchNormalization(name='block3_batchnorm_ascad_desync50')(x)
        x = AveragePooling1D(4, strides=4, name='block3_pool_ascad_desync50')(x)

        x = Flatten(name='flatten_ascad_desync50')(x)

        # Classification part
        x = Dense(15, activation='selu', name='fc1_ascad_desync50', **kwargs)(x)
        x = Dense(15, activation='selu', name='fc2_ascad_desync50', **kwargs)(x)
        x = Dense(15, activation='selu', name='fc3_ascad_desync50', **kwargs)(x)

        scores = Dense(self.code_length, activation=None, name='scores_lenet_5_rect',
                       kernel_regularizer=self.kernel_regularizer)(x)
        predictions = Activation('sigmoid', name='predictions_lenet_5_rect')(scores)

        model = Model(img_input, predictions, name='cnn_lenet_2d')
        scoring_model = Model(img_input, scores, name='cnn_lenet_2d')

        # optimizer = RMSprop(learning_rate=0.00001)
        # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model, scoring_model

    def fit(self, X, y, epochs=10, batch_size=500, verbose=1, max_lr=1e-2, **kwargs):
        return super().fit_lr(X=X, y=y, epochs=epochs, batch_size=batch_size, verbose=verbose, max_lr=max_lr, **kwargs)

    def predict_scores_old(self, X, verbose=0, **kwargs):
        X, _ = self.reshape_inputs(X, None)
        predictions1 = self.model.predict(X, verbose, **kwargs) + 1e-5
        predictions0 = 1 - predictions1
        n_test = X.shape[0]
        scores = np.zeros((n_test, 256))
        for i, code_block in enumerate(self.ecoc_code_block):
            idx = np.where(code_block == 1)[0]
            product_scores = np.zeros(n_test)
            if len(idx) > 0:
                product_scores = product_scores + np.sum(np.log(predictions1[:, idx]), axis=1)
            idx = np.where(code_block == 0)[0]
            if len(idx) > 0:
                product_scores = product_scores + np.sum(np.log(predictions0[:, idx]), axis=1)
            scores[:, i] = product_scores

        mini = -float(np.min(scores) + 10)
        self.logger.info("Minimum offset negated: {}".format(mini))
        scores = 10 ** (scores + mini)
        scores = (scores / scores.sum(axis=1)[:, np.newaxis]) * 10
        self.logger.info("Scores: \n{}".format(scores))
        return scores

    def predict_scores(self, X, verbose=0, **kwargs):
        X, _ = self.reshape_inputs(X, None)
        predictions = self.model.predict(X, verbose, **kwargs) + 1e-10
        scores = euclidean_distances(predictions, self.ecoc_code_block)
        scores = RobustScaler().fit_transform(scores.T).T
        scores = 1.0 / (1.0 + np.exp(-scores))
        self.logger.info("Scores: \n{}".format(scores))
        return scores

    def decision_function(self, X, verbose=0, **kwargs):
        return self.predict_scores(X=X, verbose=verbose, **kwargs)

    def predict_proba(self, X, verbose=0, **kwargs):
        scores = self.predict_scores(X=X, verbose=verbose, **kwargs)
        return scores

    def evaluate(self, X, y, verbose=1, **kwargs):
        return super().evaluate(X, y, verbose, **kwargs)

    def summary(self, **kwargs):
        super().summary(**kwargs)
