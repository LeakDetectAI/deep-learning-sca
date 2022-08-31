import numpy as np

from deepscapy.constants import *
from deepscapy.dataset_reader import *
from deepscapy.losses.loss_functions import *
from deepscapy.models import *

__all__ = ['LF_EXTENSION', 'model_dictionary', 'cnn_rkl_baseline_dictionary', 'datasets',
           'loss_dictionary_train_models', 'loss_dictionary_attack_models', 'loss_dictionary_rkl_models',
           'focal_loss_dict', 'rkl_loss_dict']

LF_EXTENSION = {CATEGORICAL_CROSSENTROPY_LOSS: 'CCE',
                RANKING_LOSS: 'RKL',
                CROSS_ENTROPY_RATIO: 'CER',
                FOCAL_LOSS_BE: 'FLBCE',
                FOCAL_LOSS_CE: 'FLCCE',
                FOCAL_LOSS_BER: 'FLBCER',
                FOCAL_LOSS_CER: 'FLCCER'}

model_dictionary = {ASCAD_CNN_BASELINE: ASCADCNNBaseline,
                    ASCAD_MLP_BASELINE: ASCADMLPBaseline,
                    CUSTOM_LENET5: CustomLeNet5,
                    CUSTOM_LENET5_RECT: CustomLeNet5Rectangle,
                    CUSTOM_LENET5_SQUARE: CustomLeNet5Square,
                    NAS_BASIC2: NASBasic2,
                    NAS_BASIC3: NASBasic3,
                    NAS_BASIC4: NASBasic4,
                    NAS_BASIC5: NASBasic5}

cnn_rkl_baseline_dictionary = {ASCAD_DESYNC0: CNNRankingLossASCADDesync0Baseline,
                               ASCAD_DESYNC50: CNNRankingLossASCADDesync50Baseline,
                               ASCAD_DESYNC100: CNNRankingLossASCADDesync100Baseline,
                               AES_HD: CNNRankingLossAESHDBaseline,
                               AES_RD: CNNCCEAESRDBaseline,
                               DP4_CONTEST: CNNCCEDPACV4Baseline,
                               CHES_CTF: CNNRankingLossASCADDesync100Baseline,
                               ASCAD_DESYNC0_VARIABLE: CNNRankingLossASCADDesync0Baseline,
                               ASCAD_DESYNC50_VARIABLE: CNNRankingLossASCADDesync50Baseline,
                               ASCAD_DESYNC100_VARIABLE: CNNRankingLossASCADDesync100Baseline,
                               }

datasets = {ASCAD_DESYNC0: ASCADDatasetReader, ASCAD_DESYNC50: ASCADDatasetReader, ASCAD_DESYNC100: ASCADDatasetReader,
            ASCAD_DESYNC0_VARIABLE: ASCADDatasetReader, ASCAD_DESYNC50_VARIABLE: ASCADDatasetReader,
            ASCAD_DESYNC100_VARIABLE: ASCADDatasetReader, CHES_CTF: CHESCTFDatasetReader, AES_HD: AESHDDatasetReader,
            AES_RD: AESRDDatasetReader, DP4_CONTEST: DP4ContestDatasetReader}


# if "pc2" in os.environ["HOME"]:
#     ABSOLUTE_PATH = os.path.join(os.environ["PFS_FOLDER"], "deep-learning-sca")
# else:
#     dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#     ABSOLUTE_PATH = os.path.dirname(dirname)

# Define paths
# TRAINED_MODELS = os.path.join(ABSOLUTE_PATH, "deepscapy", "trained_models")
# RESULTS = os.path.join(ABSOLUTE_PATH, "results")

# print(TRAINED_MODELS)

# FOCAL_LOSS: sigmoid_focal_loss,
loss_dictionary_train_models = {CATEGORICAL_CROSSENTROPY_LOSS: CATEGORICAL_CROSSENTROPY_LOSS,
                                CROSS_ENTROPY_RATIO: cross_entropy_ratio(),
                                # RANKING_LOSS: ranking_loss(),
                                DICE_BCE_LOSS: bce_dice_loss(),
                                FOCAL_LOSS_BE: binary_crossentropy_focal_loss(),
                                FOCAL_LOSS_CE: categorical_crossentropy_focal_loss(),
                                FOCAL_LOSS_BER: binary_crossentropy_focal_loss_ratio(),
                                FOCAL_LOSS_CER: categorical_crossentropy_focal_loss_ratio()
                                }

loss_dictionary_hpo_models = {CATEGORICAL_CROSSENTROPY_LOSS: CATEGORICAL_CROSSENTROPY_LOSS,
                              CROSS_ENTROPY_RATIO: cross_entropy_ratio(),
                              RANKING_LOSS: ranking_loss_optimized(),
                              DICE_BCE_LOSS: bce_dice_loss(),
                              FOCAL_LOSS_BE: binary_crossentropy_focal_loss(),
                              FOCAL_LOSS_CE: categorical_crossentropy_focal_loss()}

loss_dictionary_attack_models = {RANKING_LOSS: ranking_loss_optimized,
                                 FOCAL_LOSS_BE: binary_crossentropy_focal_loss,
                                 FOCAL_LOSS_CE: categorical_crossentropy_focal_loss}

loss_dictionary_rkl_models = {ASCAD_DESYNC0: ranking_loss_optimized(alpha_value=0.5),
                              ASCAD_DESYNC0_VARIABLE: ranking_loss_optimized(alpha_value=0.5),
                              ASCAD_DESYNC50: ranking_loss_optimized(alpha_value=0.5),
                              ASCAD_DESYNC50_VARIABLE: ranking_loss_optimized(alpha_value=0.5),
                              ASCAD_DESYNC100: ranking_loss_optimized(alpha_value=2),
                              ASCAD_DESYNC100_VARIABLE: ranking_loss_optimized(alpha_value=2),
                              AES_HD: ranking_loss_optimized(alpha_value=1),
                              AES_RD: ranking_loss_optimized(alpha_value=1),
                              CHES_CTF: ranking_loss_optimized(alpha_value=1),
                              DP4_CONTEST: ranking_loss_optimized(alpha_value=1)}

# trained_models_path_dict = {True: get_trained_models_path(folder=TRAINED_MODELS_TUNED),
#                             False: get_trained_models_path(folder=TRAINED_MODELS_NON_TUNED)}

hp_dict = {'optimizer': ['rmsprop', 'sgd', 'adam', 'adagrad'],
           'learning_rate': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
           'reg_strength': [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
           'kernal_initializer': ['he_uniform', 'he_normal']}

focal_loss_dict = {'alpha': [0.1, 0.25, 0.5, 0.75, 0.90],
                   'gamma': [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]}

rkl_loss_dict = {'alpha_value': list(np.linspace(0.0001, 2, num=30))}
