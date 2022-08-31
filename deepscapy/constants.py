import numpy as np

__all__ = ['CATEGORICAL_CROSSENTROPY_LOSS', 'FOCAL_LOSS_BE', 'FOCAL_LOSS_CE', 'DICE_BCE_LOSS', 'CROSS_ENTROPY_RATIO',
           'RANKING_LOSS', 'SMOOTH_TOPK_LOSS', 'HARD_TOPK_LOSS', 'SMOOTH_HARD_TOPK_LOSS', 'FOCAL_LOSS_CER',
           'FOCAL_LOSS_BER', 'ASCAD_MLP_BASELINE', 'ASCAD_CNN_BASELINE', 'CNN_ZAID_BASELINE',
           'ASCAD_DESYNC0_CNN_RKL_BASELINE', 'ASCAD_DESYNC0_CNN_RKL_BASELINE', 'ASCAD_DESYNC50_CNN_RKL_BASELINE',
           'ASCAD_DESYNC100_CNN_RKL_BASELINE', 'AES_HD_CNN_RKL_BASELINE', 'CUSTOM_LENET5', 'CUSTOM_LENET5_NEW',
           'CUSTOM_LENET5_RECT', 'CUSTOM_LENET5_SQUARE', "NAS_MODELS", 'ASCAD', 'ASCAD_DESYNC0', 'ASCAD_DESYNC50',
           'ASCAD_DESYNC0_VARIABLE', 'ASCAD_DESYNC50_VARIABLE', 'ASCAD_DESYNC100_VARIABLE', 'ASCAD_DESYNC100',
           'AES_HD', 'AES_RD', 'DP4_CONTEST', 'ASCAD_DATASETS', 'CHES_CTF', 'HYPERBAND_TUNER', 'RANDOM_TUNER',
           'BAYESIAN_TUNER', 'GREEDY_TUNER', 'TUNER_TYPES', 'MEAN_RANKS', 'GUESS_ENTROPY', 'SCORES', 'MEAN_RANK_FINAL',
           'ACCURACY', 'MIN_COMPLETE_MEAN_RANK', 'LAST_INDEX_MEAN_RANK', 'MIN_LAST_100_MEAN_RANK',
           'NAS_TRIALS_DIRECTORY', 'NAS_TRIALS_DIRECTORY_NEW', 'TRAINED_MODELS_NON_TUNED', 'TRAINED_MODELS_TUNED',
           'TRAINED_MODELS_NAS', 'TRAINED_MODELS_NAS_NEW', "RESULTS", "RESULTS_NEW", 'BASELINES', 'TRAINED_MODELS',
           'TRAINED_HYPERMODEL', 'HP_TRAINED_MODELS_TUNED', 'ASCAD_AES_Sbox', 'AES_RD_Sbox', 'AES_Sbox_inv',
           'DP4CONTEST_BOX', 'aes_hd_key', 'package_name', 'NAS_BASIC2', 'NAS_BASIC3', 'NAS_BASIC4', 'NAS_BASIC5',
           'TWOD_CNN_RECT', 'TWOD_CNN_SQR', 'ONED_CNN', 'RESHAPE_TYPES', 'INPUT_SHAPE_DICT',
           'MLP', 'TRAINABLE_PARAMS', 'NON_TRAINABLE_PARAMS', 'TOTAL_PARAMS', 'HW', 'ID']

# Losses
CATEGORICAL_CROSSENTROPY_LOSS = 'categorical_crossentropy'
FOCAL_LOSS_BE = 'sigmoid_focal_binary_crossentropy'
FOCAL_LOSS_CE = 'sigmoid_focal_categorical_crossentropy'
FOCAL_LOSS_CER = 'sigmoid_focal_categorical_crossentropy_ratio'
FOCAL_LOSS_BER = 'sigmoid_focal_binary_crossentropy_ratio'

DICE_BCE_LOSS = 'dice_bce_loss'
CROSS_ENTROPY_RATIO = 'cross_entropy_ratio'
RANKING_LOSS = 'ranking_loss'
SMOOTH_TOPK_LOSS = 'smooth_topk_loss'
HARD_TOPK_LOSS = 'hard_topk_loss'
SMOOTH_HARD_TOPK_LOSS = 'smooth_hard_topk_loss'

# Models
ASCAD_MLP_BASELINE = 'ascad_mlp_baseline'
ASCAD_CNN_BASELINE = 'ascad_cnn_baseline'
CNN_ZAID_BASELINE = 'cnn_zaid_baseline'
ASCAD_DESYNC0_CNN_RKL_BASELINE = 'cnn_rkl_ascad_desync0_baseline'
ASCAD_DESYNC50_CNN_RKL_BASELINE = 'cnn_rkl_ascad_desync50_baseline'
ASCAD_DESYNC100_CNN_RKL_BASELINE = 'cnn_rkl_ascad_desync100_baseline'
AES_HD_CNN_RKL_BASELINE = 'cnn_rkl_aes_hd_baseline'
CUSTOM_LENET5 = 'lenet5'
CUSTOM_LENET5_NEW = 'lenet5_new'
CUSTOM_LENET5_RECT = 'lenet5_rect'
CUSTOM_LENET5_SQUARE = 'lenet5_square'
NAS_BASIC2 = 'nas_basic2'
NAS_BASIC3 = 'nas_basic3'
NAS_BASIC4 = 'nas_basic4'
NAS_BASIC5 = 'nas_basic5'
NAS_RESNET3 = 'nas_resnet3'
BASELINES = [ASCAD_MLP_BASELINE, ASCAD_CNN_BASELINE, CNN_ZAID_BASELINE]
NAS_MODELS = [NAS_BASIC5]
HW = 'HW'
ID = 'ID'

# Datasets
ASCAD = 'ASCAD'
ASCAD_DESYNC0 = 'ASCAD_desync0'
ASCAD_DESYNC50 = 'ASCAD_desync50'
ASCAD_DESYNC100 = 'ASCAD_desync100'
ASCAD_DESYNC0_VARIABLE = 'ASCAD_desync0_variable'
ASCAD_DESYNC50_VARIABLE = 'ASCAD_desync50_variable'
ASCAD_DESYNC100_VARIABLE = 'ASCAD_desync100_variable'
CHES_CTF = 'CHES_CTF'
AES_HD = 'AES_HD'
AES_RD = 'AES_RD'
DP4_CONTEST = 'DP4CONTEST'

ASCAD_DATASETS = [ASCAD_DESYNC0, ASCAD_DESYNC50, ASCAD_DESYNC100, ASCAD_DESYNC0_VARIABLE, ASCAD_DESYNC50_VARIABLE,
                  ASCAD_DESYNC100_VARIABLE]

# Important paths
# TRAINED_MODELS = 'trained_models'
# DATASETS = 'datasets'

# Tuner types
HYPERBAND_TUNER = 'hyperband'
RANDOM_TUNER = 'random'
BAYESIAN_TUNER = 'bayesian'
GREEDY_TUNER = 'greedy'
TUNER_TYPES = [HYPERBAND_TUNER, GREEDY_TUNER, RANDOM_TUNER, BAYESIAN_TUNER]

# Attack results file names
MEAN_RANKS = 'mean_ranks'
GUESS_ENTROPY = 'guess_entropy'
SCORES = 'scores'
MEAN_RANK_FINAL = 'mean_rank_final'
ACCURACY = 'accuracy'
MIN_COMPLETE_MEAN_RANK = 'min_complete_mean_rank'
LAST_INDEX_MEAN_RANK = 'last_index_mean_rank'
MIN_LAST_100_MEAN_RANK = 'min_last_100_mean_rank'
TRAINABLE_PARAMS = 'trainable_params'
NON_TRAINABLE_PARAMS = 'non_trainable_params'
TOTAL_PARAMS = 'total_params'

# Trained models folder names
TRAINED_MODELS = "trained_models"
TRAINED_MODELS_NON_TUNED = 'non_tuned_models'

TRAINED_HYPERMODEL = 'trained_hypermodel'
HP_TRAINED_MODELS_TUNED = 'best_hp_tuned'
TRAINED_MODELS_TUNED = 'tuned_models'

NAS_TRIALS_DIRECTORY = 'nas_trials_directory'
NAS_TRIALS_DIRECTORY_NEW = 'nas_trials_directory_new'
TRAINED_MODELS_NAS = 'nas_models'
TRAINED_MODELS_NAS_NEW = 'nas_models_new'
RESULTS = "results"
RESULTS_NEW = "results_new"

# Reshape types
TWOD_CNN_RECT = '2dCNNRect'
TWOD_CNN_SQR = '2dCNNSqr'
ONED_CNN = '1dCNN'
RESHAPE_TYPES = [TWOD_CNN_RECT, TWOD_CNN_SQR, ONED_CNN]
INPUT_SHAPE_DICT = {TWOD_CNN_RECT: 'rect', TWOD_CNN_SQR: 'sqr', ONED_CNN: 'one_d'}
MLP = 'MLP'

# Datasets Sbox
ASCAD_AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

AES_RD_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

AES_Sbox_inv = np.array([0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
                         0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
                         0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
                         0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
                         0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
                         0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
                         0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
                         0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
                         0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
                         0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
                         0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
                         0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
                         0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
                         0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
                         0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
                         0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
                         0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
                         0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
                         0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
                         0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
                         0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
                         0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
                         0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
                         0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
                         0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
                         0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
                         0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
                         0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
                         0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
                         0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
                         0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
                         0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
                         ])

DP4CONTEST_BOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

# Key
aes_hd_key = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
#aes_hd_key = [43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60]
# Miscellaneous
package_name = 'deepscapy'