import keras.backend as K


def dice_coef(y_true, y_pred, classes=20):
    smooth = 1.

    dice_all = 0
    for layer in range(classes):
        true_layer = y_true[:, :, :, layer]
        pred_layer = y_pred[:, :, :, layer]
        y_true_f = K.flatten(true_layer)
        y_pred_f = K.flatten(pred_layer)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        dice_all += dice
    return dice_all / classes


def jaccard_coef(y_true, y_pred, classes=20):
    smooth = 1e-12
    jaccard = 0
    for layer in range(classes):
        true_layer = y_true[:, :, :, layer]
        pred_layer = y_pred[:, :, :, layer]

        intersection = K.sum(true_layer * pred_layer, axis=[0, -1, -2])
        sum_ = K.sum(true_layer + pred_layer, axis=[0, -1, -2])

        jac = (intersection + smooth) / (sum_ - intersection + smooth)

        jaccard += jac
    return jaccard / classes


def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def jaccard_loss(y_true, y_pred):
    return 1. - jaccard_coef(y_true, y_pred)
