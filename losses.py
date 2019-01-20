import keras.backend as K
import tensorflow as tf
import numpy as np


def focal_loss(y_true, y_pred):
    gamma = 0.75
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def dice_coef(y_true, y_pred):
    smooth = 1.
    dice_all = 0
    for layer in range(20):
        true_layer = y_true[:, :, :, layer]
        pred_layer = y_pred[:, :, :, layer]
        y_true_f = K.flatten(true_layer)
        y_pred_f = K.flatten(pred_layer)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        dice_all += dice
    return dice_all / 20


def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    jaccard = 0
    for layer in range(20):
        true_layer = y_true[:, :, :, layer]
        pred_layer = y_pred[:, :, :, layer]

        intersection = K.sum(true_layer * pred_layer, axis=[0, -1, -2])
        sum_ = K.sum(true_layer + pred_layer, axis=[0, -1, -2])

        jac = (intersection + smooth) / (sum_ - intersection + smooth)

        jaccard += jac
    return jaccard / 20


def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def jaccard_loss(y_true, y_pred):
    return 1. - jaccard_coef(y_true, y_pred)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 20)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def custom_loss(y_true, y_pred):
    return 0.5*dice_loss(y_true, y_pred) + mean_iou(y_true, y_pred)
