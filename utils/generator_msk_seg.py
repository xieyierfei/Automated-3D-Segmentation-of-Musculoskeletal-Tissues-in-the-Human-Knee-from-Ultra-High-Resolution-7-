from __future__ import print_function, division

import numpy as np
from os import listdir
from os.path import splitext
from random import shuffle
from re import split
import h5py

# find unique data regardless of the file prefix
def calc_generator_info(data_path, batch_size, learn_files=[]):

    files = listdir(data_path)
    unique_filename = {}

    for file in files:
        file, _ = splitext(file)
        if not file in unique_filename:
            unique_filename[file] = file

    files = unique_filename.keys()

    # Set total number of files based on argument for limiting training size
    if learn_files:
        nfiles = learn_files
    else:
        nfiles = len(files)

    batches_per_epoch = nfiles // batch_size

    return(files, batches_per_epoch)

def img_generator_oai(data_path, seg_path, batch_size, img_size, tag, testing=False, shuffle_epoch=True):

    files, batches_per_epoch = calc_generator_info(data_path, batch_size)
    files = list(files)

    x = np.zeros((batch_size,)+img_size)
    y = np.zeros((batch_size,)+img_size)

    while True:

        if shuffle_epoch:
            shuffle(files)
        else:
            files = sort_files(files, tag)


        for batch_cnt in range(batches_per_epoch):
            for file_cnt in range(batch_size):

                file_ind = batch_cnt * batch_size + file_cnt
                im_path = np.load('%s/%s.npy'%(data_path, files[file_ind]))
                im = im_path.astype('float32')

                #seg_id = shuffle_id[file_ind]
                s_path = np.load('%s/%s.npy'%(seg_path, files[file_ind]))
                seg = s_path.astype('float32')

                x[file_cnt, ...,0] = im
                y[file_cnt, ...,0] = seg
                fname = files[file_ind]

            if testing is False:
                yield(x,y)
            else:
                fname = files[file_ind]

                yield(x, y, fname)


def sort_files(files, tag):

    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    file_id = [None] * len(files)
    for cnt1 in range(len(files)):
        tmp = split('_', files[cnt1])

        if (tag == 'oai'):
            tmp = int(160 * int(tmp[1]) + int(tmp[3]))
        else:
            raise ValueError('Specified tag (%s) is unsupported' % tag)

        file_id[cnt1] = int(tmp)

    order = argsort(file_id)

    return [files[cnt1] for cnt1 in order]


