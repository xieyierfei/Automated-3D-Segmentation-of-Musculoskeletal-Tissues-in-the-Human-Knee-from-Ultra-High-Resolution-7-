import numpy as np
from keras import backend as K

# Dice function loss optimizer
def dice_loss(y_true, y_pred):


    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    ovlp = K.sum(y_true*y_pred)

    mu = K.epsilon()
    dice = (2.0 * ovlp + mu) / (K.sum(y_true) + K.sum(y_pred) + mu)
    loss = 1 - dice

    return loss

# Dice function loss optimizer
# During test time since it includes a discontinuity
def dice_loss_test(y_true, y_pred):

    recon = np.squeeze(y_true)
    pred = np.squeeze(y_pred)
    y_pred = (y_pred > 0.25) * y_pred

    szp = y_pred.shape
    img_len = szp[1]*szp[2]*szp[3]

    y_true = np.reshape(y_true, (-1, img_len))
    y_pred = np.reshape(y_pred, (-1, img_len))

    ovlp = np.sum(y_true*y_pred, axis=-1)

    mu = 1e-07
    dice = (2.0 * ovlp + mu) / (np.sum(y_true, axis=-1) + np.sum(y_pred, axis=-1) + mu)

    return dice

