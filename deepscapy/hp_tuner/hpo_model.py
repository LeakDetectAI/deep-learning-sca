import os

import keras_tuner as kt
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop, Nadam, Adagrad
from keras.regularizers import l2

from deepscapy.callbacks import OneCycleLR
from deepscapy.constants import TRAINED_MODELS_TUNED
from deepscapy.models import CNNRankingLossASCADDesync0Baseline, CNNRankingLossASCADDesync50Baseline, \
    CNNRankingLossASCADDesync100Baseline, CNNRankingLossAESHDBaseline, CNNCCEAESRDBaseline
from deepscapy.utils import check_file_exists, get_trained_models_path


class HPOModel(kt.HyperModel):
    def __init__(self, learner, learner_params, hp_dict, model_name, max_lr=1e-3, **kwargs):
        super(HPOModel, self).__init__(**kwargs)
        self.learner = learner
        self.learner_params = learner_params
        self.model = None
        self.sca_model = None
        self.hp_dict = hp_dict
        self.model_name = model_name
        self.max_lr = max_lr
        # self.model_file = os.path.join(os.getcwd(), 'deepscapy', 'trained_models', '{}_{}.h5'.format(model_name, tuner_type))
        self.model_file = os.path.join(get_trained_models_path(folder=TRAINED_MODELS_TUNED), '{}.tf'.format(model_name))

    def build(self, hp):
        # Initialize hyperparameters according to the values passed in hp_dict
        optimizer = hp.Choice('optimizer', values=self.hp_dict['optimizer'])
        # learning_rate = hp.Float("learning_rate", min_value=self.hp_dict['learning_rate'][0],
        #                        max_value=self.hp_dict['learning_rate'][1], sampling="log")
        # reg_strength = hp.Float("reg_strength", min_value=self.hp_dict['reg_strength'][0],
        #                       max_value=self.hp_dict['reg_strength'][1], sampling="log")
        reg_strength = hp.Choice('reg_strength', values=self.hp_dict['reg_strength'])
        learning_rate = hp.Choice('learning_rate', values=self.hp_dict['learning_rate'])
        kernel_initializer = hp.Choice('kernal_initializer', values=self.hp_dict['kernal_initializer'])

        # Initialize the optimizer with the learning_rate
        if optimizer.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, nesterov=True, momentum=0.9)
        elif optimizer.lower() == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer.lower() == 'nadam':
            optimizer = Nadam(learning_rate=learning_rate)
        elif optimizer.lower() == 'adagrad':
            optimizer = Adagrad(learning_rate=learning_rate)
        else:
            optimizer = None

        self.learner_params['optimizer'] = optimizer
        self.learner_params['kernel_initializer'] = kernel_initializer
        self.learner_params['kernel_regularizer'] = l2(l=reg_strength)

        # if str(self.learner_params['loss_function']) == str(loss_dictionary_hpo_models[FOCAL_LOSS]):
        # alpha = hp.Choice('alpha', values=self.hp_dict['alpha'])
        # gamma = hp.Choice('gamma', values=self.hp_dict['gamma'])
        # from_logits = hp.Boolean('from_logits')
        # self.learner_params['loss_function'] = loss_dictionary_hpo_models[FOCAL_LOSS](alpha=alpha, gamma=gamma, from_logits=from_logits)
        # self.learner_params['loss_function'] = loss_dictionary_hpo_models[FOCAL_LOSS]()

        # if str(self.learner_params['loss_function']) == str(loss_dictionary_hpo_models[RANKING_LOSS]):
        # alpha_value = hp.Choice('alpha_value', values=self.hp_dict['alpha_value'])
        # self.learner_params['loss_function'] = loss_dictionary_hpo_models[RANKING_LOSS](alpha_value=alpha_value)
        # self.learner_params['loss_function'] = loss_dictionary_hpo_models[RANKING_LOSS]()

        self.sca_model = self.learner(**self.learner_params)
        return self.sca_model.model

    def fit(self, hp, model, *args, **kwargs):
        callbacks = kwargs.pop('callbacks', [])
        batch_size = kwargs.pop('batch_size', 200)
        check_file_exists(os.path.dirname(self.model_file))
        save_model = ModelCheckpoint(self.model_file)
        callbacks.append(save_model)
        samples = args[0][0].shape[0]
        if self.learner in [CNNRankingLossASCADDesync0Baseline, CNNRankingLossASCADDesync50Baseline,
                            CNNRankingLossASCADDesync100Baseline, CNNRankingLossAESHDBaseline, CNNCCEAESRDBaseline]:
            lr_manager = OneCycleLR(max_lr=self.max_lr, batch_size=batch_size, samples=samples, end_percentage=0.2,
                                    scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None, verbose=True)
            callbacks.append(lr_manager)
        kwargs['callbacks'] = callbacks
        return model.fit(*args, **kwargs)
