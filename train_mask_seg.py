from __future__ import print_function, division

import numpy as np
import pickle
import math
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.optimizer_v2 import adam
from keras import backend as K
import keras.callbacks as kc

from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import LambdaCallback as lcb
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import TensorBoard as tfb
from keras.optimizers import adam_v2

from utils.generator_msk_seg import calc_generator_info, img_generator_oai
from utils.models import build_unet
from utils.losses import dice_loss

# Training and validation data location
train_im_path = 'D:/high resolution/train/short_im_npy/'
train_seg_path = 'D:/high resolution/train/short_seg_npy/'
valid_im_path = 'D:/high resolution/valid/valid_im_npy/'
valid_seg_path = 'D:/high resolution/valid/valid_seg_npy/'
dir_plot_save = 'D:/high resolution/checkpoint/'
#test_im_path = '/rds/general/user/gg221/home/OAI/test_imgs_only/test_slices_im/'
#test_seg_path = '/rds/general/user/gg221/home/OAI/test-ground-truth/test_slices_seg/'
train_batch_size = 1
valid_batch_size = 1

# Locations and names for saving training checkpoints
cp_save_path = 'D:/high resolution/weights/'
cp_save_tag = 'unet_2d_men'
pik_save_path = 'D:/high resolution/checkpoint/' + cp_save_tag + '.dat'

# Model parameters
n_epochs = 5
file_types = ['png']
# Tissues are in the following order
# 0. Femoral 1. Lat Tib 2. Med Tib 3. Pat 4. Lat Men 5. Med Men
tissue = np.arange(0, 6)
# Load pre_trained model
model_weights = 'D:/OAI/weights/unet_2d_men_weights.015-0.3429.h5'

# Training and validation image size
img_size = (512,512, len(file_types))
# What dataset are we training on? 'dess' or 'oai'


# Restrict number of files learned. Default is all []
learn_files = []


# Freeze layers in transfer learning
layers_to_freeze = range(0,67)

# learning rate schedule
# Implementing a step dacay for now
def step_decay(epoch):
    initial_lrate = 1e-4
    drop = 0.8
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def train_seg(img_size, train_im_path, train_seg_path, valid_im_path, valid_seg_path, train_batch_size,
              valid_batch_size,
              cp_save_path, cp_save_tag, n_epochs, file_types, pik_save_path, learn_files):
    # set image format to be (N, dim1, dim2, dim3, ch)
    #    K.set_image_data_format('channels_last')
    train_files, train_nbatches = calc_generator_info(train_im_path, train_batch_size, learn_files)
    valid_files, valid_nbatches = calc_generator_info(valid_im_path, valid_batch_size)

    # Print some useful debugging information
    print('INFO: Train size: %d, batch size: %d' % (len(train_files), train_batch_size))
    print('INFO: Valid size: %d, batch size: %d' % (len(valid_files), valid_batch_size))
    print('INFO: Image size: %s' % (img_size,))
    print('INFO: Image types included in training: %s' % (file_types,))
    print('INFO: Number of tissues being segmented: %d' % len(tissue))

    # create the unet model
    model = build_unet(img_size, n_classes=1)
    if model_weights is not None:
        model.load_weights(model_weights, by_name=True)

    # Set up the optimizer
    model.compile(optimizer=adam_v2.Adam(learning_rate=1e-9, beta_1=0.99, beta_2=0.995, epsilon=1e-08, decay=0.0),
                  loss=dice_loss)

    # Optinal, but this allows you to freeze layers if you want for transfer learning
    for lyr in layers_to_freeze:
        model.layers[lyr].trainable = False

    print(model.summary())
    print('trainable:')
    for x in model.trainable_variables:
        print(x.name)
    print('\n')
    # model callbacks per epoch
    cp_cb = ModelCheckpoint(cp_save_path + '/' + cp_save_tag + '_weights.{epoch:03d}-{val_loss:.4f}.h5',
                            save_best_only=True)
    tfb_cb = tfb('D:/high resolution/tf_log/',
                 histogram_freq=1,
                 write_grads=False,
                 write_images=False)
    lr_cb = lrs(step_decay)
    hist_cb = LossHistory()

    callbacks_list = [tfb_cb, cp_cb, hist_cb, lr_cb]

    # Start the training
    history = model.fit_generator(
        img_generator_oai(train_im_path, train_seg_path, train_batch_size, img_size, 'train'),
        train_nbatches,
        epochs=n_epochs,
        verbose=2,
        validation_data=img_generator_oai(valid_im_path, valid_seg_path, valid_batch_size, img_size, 'valid'),
        validation_steps=valid_nbatches,
        callbacks=callbacks_list)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)

    plt.legend()
    plt.show()
    plt.savefig(os.path.join(dir_plot_save, 'model_loss.png'))
    
    # Save files to write as output
    data = [hist_cb.epoch, hist_cb.lr, hist_cb.losses, hist_cb.val_losses]
    with open(pik_save_path, "wb") as f:
        pickle.dump(data, f)

    return hist_cb


# Print and asve the training history
class LossHistory(kc.Callback):
    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.losses = []
        self.lr = []
        self.epoch = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        self.epoch.append(len(self.losses))


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA DEVICE ORDER"] = "PCI BUS ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = "0"

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    model = build_unet(img_size, n_classes=1)
    #print(model.summary())
    train_seg(img_size, train_im_path, train_seg_path, valid_im_path, valid_seg_path, train_batch_size,
              valid_batch_size,
              cp_save_path, cp_save_tag, n_epochs, file_types, pik_save_path, learn_files)


