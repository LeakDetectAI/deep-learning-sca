from keras import backend as K


# Source: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
def generalized_dice_coefficient(y_true, y_pred, smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (
            K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


# Source: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions
def dice_loss(y_true, y_pred, smooth):
    loss = 1 - generalized_dice_coefficient(y_true, y_pred, smooth)
    return loss
