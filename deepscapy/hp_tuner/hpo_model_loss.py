import os

import keras_tuner as kt
from keras.callbacks import ModelCheckpoint

from deepscapy.callbacks import OneCycleLR
from deepscapy.constants import *
from deepscapy.experimental_utils import loss_dictionary_attack_models
from deepscapy.models import *
from deepscapy.utils import get_trained_models_path, check_file_exists


class HPOModelLoss(kt.HyperModel):
    def __init__(self, learner, learner_params, hp_dict, lf_name, max_lr=1e-3, **kwargs):
        super(HPOModelLoss, self).__init__(**kwargs)
        self.learner = learner
        self.learner_params = learner_params
        self.model = None
        self.sca_model = None
        self.hp_dict = hp_dict
        self.max_lr = max_lr
        self.lf_name = lf_name
        model_name = learner_params['model_name']
        # self.model_file = os.path.join(os.getcwd(), 'deepscapy', 'trained_models', '{}_{}.h5'.format(model_name, tuner_type))
        self.model_file = os.path.join(get_trained_models_path(folder=TRAINED_MODELS_TUNED), '{}.tf'.format(model_name))

    def build(self, hp):
        nb_class = self.learner_params['num_classes']
        if self.lf_name in [FOCAL_LOSS_BE, FOCAL_LOSS_CE]:
            alpha = hp.Choice('alpha', values=self.hp_dict['alpha'])
            gamma = hp.Choice('gamma', values=self.hp_dict['gamma'])
            loss_params = dict(alpha=alpha, gamma=gamma)

        elif self.lf_name == RANKING_LOSS:
            alpha_value = hp.Choice('alpha_value', values=self.hp_dict['alpha_value'])
            loss_params = dict(alpha_value=alpha_value)
        print(loss_params)
        self.learner_params['loss_function'] = loss_dictionary_attack_models[self.lf_name](**loss_params)
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


class HPOModelTuner(kt.RandomSearch):
    def __init__(self, **kwargs):
        self.learner = kwargs['hypermodel'].learner
        self.max_lr = kwargs['hypermodel'].max_lr
        super(HPOModelTuner, self).__init__(**kwargs)
