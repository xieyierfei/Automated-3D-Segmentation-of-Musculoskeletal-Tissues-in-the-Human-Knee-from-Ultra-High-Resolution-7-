from __future__ import print_function, division

import numpy as np
import h5py
import time
import os
import tensorflow as tf
from keras import backend as K

from utils.generator_msk_seg import calc_generator_info, img_generator_oai
from utils.models import build_unet
from utils.losses import dice_loss_test
import utils.utils_msk_seg as segutils

# Specify directories
test_result_path = 'D:/high resolution/train/im_npy/'
test_path = 'D:/high resolution/train/im_npy/'
seg_path = 'D:/high resolution/train/seg_npy/'



# Parameters for the model testing
img_size = (384,384,1)
seg_size = (384,384,1)
file_types = ['im']
test_batch_size = 1
save_file = False
tag = 'oai'

# Test with pre_trained weights
model_weights = '/rds/general/user/gg221/home/OAI/weights/unet_2d_men_weights.020-0.3430.h5'

def test_seg(test_result_path, test_path, seg_path, img_size,
             test_batch_size, save_file, model_weights):

    img_cnt = 0

    # set image format to be (N, dim1, dim2, dim3, ch)
    #K.set_image_data_format('channels_last')

    # create the unet model
    model = build_unet(img_size, n_classes=1)
    model.load_weights(model_weights)

    # All of the testing currently assumes that there is a reference to test against.
    # Comment out these lines if testing on reference-les data
    dice_losses = np.array([])
    cv_values = np.array([])
    voe_values = np.array([])
    vd_values = np.array([])

    start = time.time()

    # Read the files that will be segmented
    test_files, ntest = calc_generator_info(test_path, test_batch_size)
    print('INFO: Test size: %d, Number of batches: %d' % (len(test_files), ntest))

    # Iterature through the files to be segmented
    for x_test, y_test, fname in img_generator_oai(test_path, seg_path, test_batch_size,
                                                   img_size, tag,
                                                   testing=True):

        # Perform the actual segmentation using pre_loaded model
        recon = model.predict(x_test, batch_size= test_batch_size)

        # Calculate real time metrics
        dl = np.mean(segutils.calc_dice(recon, y_test))
        dice_losses = np.append(dice_losses, dl)

        cv = np.mean(segutils.calc_cv(recon, y_test))
        cv_values = np.append(cv_values, cv)

        voe = np.mean(segutils.calc_voe(y_test, recon))
        voe_values = np.append(voe_values, voe)

        vd = np.mean(segutils.calc_vd(y_test, recon))
        vd_values = np.append(vd_values, vd)

        #print(' Image #%0.2d (%s). Dice = %0.3f CV = %2.1f VOE = %2.1f VD = %2.1f' % (img_cnt, fname[0:11], dl, cv, voe, vd))

        # Write output file per batch
        if save_file is True:
            save_name = '%s/%s.pred' %(test_result_path, fname)
            with h5py.File(save_name, 'w') as h5f:
                h5f.create_dataset('recon', data=recon)

        img_cnt += 1
        if img_cnt == ntest:
            break

    end = time.time()

    # Print some summary statistics
    print('--'*20)
    print('Overall Summary:')
    print('Dice Mean= %0.4f Std = %0.3f'   % (np.mean(dice_losses), np.std(dice_losses)))
    print('CV Mean = %0.4f Std = %0.3f'    % (np.mean(cv_values), np.std(cv_values)))
    print('VOE Mean = %0.4f Std = %0.3f'   % (np.mean(voe_values), np.std(voe_values)))
    print('VD Mean = %0.4f Std = %0.3f'    % (np.mean(vd_values), np.std(vd_values)))
    print('Time required = %0.1f seconds.' % (end-start))
    print('--'*20)

if __name__ == '__main__':

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    test_seg(test_result_path, test_path, seg_path, img_size,
             test_batch_size, save_file, model_weights)