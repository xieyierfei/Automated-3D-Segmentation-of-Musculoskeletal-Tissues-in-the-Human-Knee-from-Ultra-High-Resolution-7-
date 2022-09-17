from __future__  import print_function, division

import numpy as np
import pickle
import math
import os
import keras

from keras.optimizer_v2 import adam
from keras import backend as K
import keras.callbacks as kc
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import LambdaCallback as lcb
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import TensorBoard as tfb
from keras.optimizers import adam_v2

from utils.generator_msk_seg import calc_generator_info, img_generator_oai
from utils.models import build_unet
from utils.losses import dice_loss

# Training and validation data location
train_im_path = 'D:/OAI/train/short_train_slices_im/'
train_seg_path = 'D:/OAI/train/short_train_slices_seg/'
valid_im_path = 'D:/OAI/valid/short_valid_slices_im/'
valid_seg_path = 'D:/OAI/valid/short_valid_slices_seg/'
test_im_path = 'D:/OAI/test_imgs_only/test_slices_im/'
test_seg_path = 'D:/OAI/test-ground-truth/test_slices_seg/'
dir_plot_save = 'D:/OAI/checkpoint/'
train_batch_size = 5
valid_batch_size = 5

# Locations and names for saving training checkpoints
cp_save_path = 'D:/OAI/weights/'
cp_save_tag = 'unet_2d_men'
pik_save_path = 'D:/OAI/checkpoint/' + cp_save_tag + '.dat'

# Model parameters
n_epochs = 10
file_types = ['im']
# Tissues are in the following order
# 0. Femoral 1. Lat Tib 2. Med Tib 3. Pat 4. Lat Men 5. Med Men
tissue = np.arange(0,6)
# Load pre_trained model
##### model_weights = 'D:/OAI/weights/'

# Training and validation image size
img_size = (512, 512, len(file_types))
# What dataset are we training on? 'dess' or 'oai'
tag = 'oai'

# Restrict number of files learned. Default is all []
learn_files = []
# Freeze layers in transfer learning
##### layers_to_freeze = []

# learning rate schedule
# Implementing a step dacay for now
def step_decay(epoch):
    initial_lrate = 1e-3
    drop = 0.8
    epochs_drop = 1.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

def train_seg(img_size, train_im_path, train_seg_path, valid_im_path, valid_seg_path, train_batch_size, valid_batch_size,
              cp_save_path, cp_save_tag, n_epochs, file_types, pik_save_path,
              tag, learn_files):

    # set image format to be (N, dim1, dim2, dim3, ch)
#    K.set_image_data_format('channels_last')
    image_gen_args = dict(
        rescale=1. / 255,
        rotation_range=45,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        height_shift_range=0.2,
        fill_mode='constant'
    )
    mask_gen_args = dict(
        # rescale= 1. / 255,
        rotation_range=45,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.2,
        height_shift_range=0.2,
        fill_mode='constant',
        preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype))
    val_image_args = dict(
        rescale=1. / 255)
    val_mask_args = dict(
        # rescale= 1. / 255,
        preprocessing_function=lambda x: np.where(x > 0, 1, 0).astype(x.dtype))

    image_datagen = keras.preprocessing.image.ImageDataGenerator(**image_gen_args)
    mask_datagen = keras.preprocessing.image.ImageDataGenerator(**mask_gen_args)
    val_image_datagen = keras.preprocessing.image.ImageDataGenerator(**val_image_args)
    val_mask_datagen = keras.preprocessing.image.ImageDataGenerator(**val_mask_args)
    seed = 1

    image_generator = image_datagen.flow_from_directory(
        train_im_path,
        class_mode=None,
        target_size=(384, 384),
        color_mode="grayscale",
        batch_size=4,
        shuffle=False,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_seg_path,
        class_mode=None,
        target_size=(384, 384),
        color_mode="grayscale",
        batch_size=4,
        shuffle=False,
        seed=seed)
    train_gen = zip(image_generator, mask_generator)
    val_image_generator = val_image_datagen.flow_from_directory(
        valid_im_path,
        class_mode=None,
        target_size=(384, 384),
        color_mode="grayscale",
        batch_size=4,
        shuffle=False,
        seed=seed)
    val_mask_generator = val_mask_datagen.flow_from_directory(
        valid_seg_path,
        class_mode=None,
        target_size=(384, 384),
        color_mode="grayscale",
        batch_size=4,
        shuffle=False,
        seed=seed)
    val_gen = zip(val_image_generator, val_mask_generator)
    train_files, train_nbatches = calc_generator_info(train_im_path, train_batch_size, learn_files)
    valid_files, valid_nbatches = calc_generator_info(valid_im_path, valid_batch_size)

    # Print some useful debugging information
    #print('INFO: Train size: %d, batch size: %d' % (len(train_files), train_batch_size))
    #print('INFO: Valid size: %d, batch size: %d' % (len(valid_files), valid_batch_size))
    #print('INFO: Image size: %s' % (img_size,))
    #print('INFO: Image types included in training: %s' % (file_types,))
    #print('INFO: Number of tissues being segmented: %d' % len(tissue))

    # create the unet model
    model = build_unet(img_size, n_classes=1)
#####    if model_weights is not None:
#####        model.load_weights(model_weights, by_name=True)

    # Set up the optimizer
    model.compile(optimizer=adam_v2.Adam(learning_rate=1e-4, beta_1=0.99, beta_2=0.995, epsilon=1e-08, decay=0.0),
                  loss=dice_loss)

    # Optinal, but this allows you to freeze layers if you want for transfer learning
#####    for lyr in layers_to_freeze:
#####        model.layers[lyr].trainable = False

    # model callbacks per epoch
    cp_cb = ModelCheckpoint(cp_save_path + '/' + cp_save_tag + '_weights.{epoch:03d}-{val_loss:.4f}.h5', save_best_only=True)
    tfb_cb = tfb('D:/OAI/tf_log/',
                 histogram_freq=1,
                 write_grads=False,
                 write_images=False)
    lr_cb = lrs(step_decay)
    hist_cb = LossHistory()

    callbacks_list = [tfb_cb, cp_cb, hist_cb, lr_cb]

    # Start the training
    history = model.fit(
        train_gen,
        steps_per_epoch=train_nbatches,
        validation_data=val_gen,
        validation_steps=valid_nbatches,
        verbose=1,
        epochs=n_epochs,
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
        pickle.dump(data,f)

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

    model = build_unet(img_size, n_classes=1)
    #print(model.summary())
    train_seg(img_size, train_im_path, train_seg_path, valid_im_path, valid_seg_path, train_batch_size, valid_batch_size,
              cp_save_path, cp_save_tag, n_epochs, file_types, pik_save_path,
              tag, learn_files)
