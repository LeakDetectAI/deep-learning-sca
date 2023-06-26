import logging
import math
import os

import numpy as np
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow_addons.callbacks import AverageModelCheckpoint
from tensorflow_addons.optimizers import SWA

from deepscapy.callbacks import OneCycleLR
from deepscapy.constants import ONED_CNN, TWOD_CNN_RECT, TWOD_CNN_SQR, MLP, TRAINED_MODELS_TUNED
from deepscapy.core.sca_base_model import SCABaseModel
from deepscapy.utils import check_file_exists
from deepscapy.utils import get_trained_models_path

__all__ = ['SCANNModel']

class SCANNModel(SCABaseModel):
    def __init__(self, model_name, num_classes, input_dim, model_type, loss_function='categorical_crossentropy',
                 kernel_regularizer=None, kernel_initializer="he_uniform", optimizer=RMSprop(learning_rate=0.00001),
                 metrics=['accuracy'], weight_averaging=False, **kwargs):

        self.num_classes = num_classes
        self.classes_ = np.arange(num_classes)
        self.input_dim = input_dim
        self.loss_function = loss_function
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.optimizer = optimizer
        self.metrics = metrics
        self.model_type = model_type
        self.weight_averaging = weight_averaging
        self.model_name = model_name
        self.logger = logging.getLogger(SCANNModel.__name__)
        if self.model_type not in [ONED_CNN, TWOD_CNN_RECT, TWOD_CNN_SQR, MLP]:
            raise Warning('Input reshaping not defined for the specified model type {}'.format(model_type))
        # check the model path, make the default one in the deep-learning-sca, fileformat dataset_type_model_lf
        if '_tuned' not in self.model_name:
            self.model_file = os.path.join(get_trained_models_path(), '{}.tf'.format(self.model_name))
        else:
            self.model_file = os.path.join(get_trained_models_path(TRAINED_MODELS_TUNED), '{}.tf'.format(self.model_name))

        self.model, self.scoring_model = self._construct_model_(kernel_regularizer=self.kernel_regularizer,
                                                                kernel_initializer=self.kernel_initializer)
        if self.weight_averaging:
            self.model.compile(loss=self.loss_function, optimizer=SWA(self.optimizer), metrics=self.metrics)
            self.scoring_model.compile(loss=self.loss_function, optimizer=SWA(self.optimizer), metrics=self.metrics)
        else:
            self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)
            self.scoring_model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=self.metrics)


    def _construct_model_(self, **kwargs):
        raise NotImplemented

    def reshape_inputs(self, X, y):
        if self.model_type == ONED_CNN:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        if self.model_type == TWOD_CNN_RECT:
            s1 = self.input_dim[0]
            s2 = self.input_dim[1]
            X = X.reshape(-1, s1, s2, 1)
        if self.model_type == TWOD_CNN_SQR:
            s1 = self.input_dim[0]
            s2 = self.input_dim[1]
            assert s1 == s2, "Both dimensions should be same for a squared input"
            assert s1 == int(math.sqrt(X.shape[1])), "Input dimension cannot be formed, the sqrt"
            X = X.reshape(-1, s1, s2, 1)
        if y is not None:
            y = to_categorical(y, num_classes=self.num_classes)
        return X, y

    def fit(self, X, y, epochs=200, batch_size=100, verbose=1, **kwargs):
        X, y = self.reshape_inputs(X, y)
        check_file_exists(os.path.dirname(self.model_file))
        if not self.weight_averaging:
            save_model = ModelCheckpoint(self.model_file)
        else:
            save_model = AverageModelCheckpoint(filepath=self.model_file, update_weights=True)
        callbacks = [save_model]
        self.logger.info(dict(batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose))
        self.logger.info(dict(X=X.shape, y=y.shape))
        if 'ranking_loss_optimized' in str(self.loss_function):
            self.scoring_model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose,
                                   **kwargs)
            self.model.save(self.model_file, overwrite=True)
        else:
            self.model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose,
                           **kwargs)
        return self

    def fit_lr(self, X, y, epochs=200, batch_size=100, max_lr=1e-3, verbose=0, **kwargs):
        X, y = self.reshape_inputs(X, y)
        check_file_exists(os.path.dirname(self.model_file))
        if not self.weight_averaging:
            save_model = ModelCheckpoint(self.model_file)
        else:
            save_model = AverageModelCheckpoint(filepath=self.model_file, update_weights=True)
        # This doesn't work for now, check later to fix the issue, update 19.02.2022 issue resolved
        lr_manager = OneCycleLR(max_lr=max_lr, batch_size=batch_size, samples=X.shape[0], end_percentage=0.2,
                                scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None, verbose=True)
        callbacks = [save_model, lr_manager]
        self.logger.info(dict(batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose))
        self.logger.info(dict(X=X.shape, y=y.shape))
        if 'ranking_loss_optimized' in str(self.loss_function):
            self.scoring_model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose,
                                   **kwargs)
            self.model.save(self.model_file, overwrite=True)
        else:
            self.model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=verbose,
                           **kwargs)
        return self

    def predict(self, X, verbose=0, **kwargs):
        X, _ = self.reshape_inputs(X, None)
        scores = self.predict_scores(X=X, verbose=verbose, **kwargs)
        pred = np.argmax(scores, axis=1)
        return pred

    def predict_scores(self, X, verbose=0, batch_size=200, **kwargs):
        X, _ = self.reshape_inputs(X, None)
        predictions = self.model.predict(x=X, batch_size=batch_size, verbose=verbose, **kwargs)
        return predictions

    def evaluate(self, X, y, verbose=1, batch_size=200, **kwargs):
        X, y = self.reshape_inputs(X, y)
        model_metrics = self.model.evaluate(x=X, y=y, batch_size=batch_size, verbose=verbose, **kwargs)
        return model_metrics

    def summary(self, **kwargs):
        self.model.summary(**kwargs)

