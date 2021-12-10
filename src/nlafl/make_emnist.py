"""
Create the EMNIST dataset numpy files by extracting the images from .h5 files.
"""

import numpy as np
import h5py

# from src.fed_learning_keras import common
import common

# Read the .h5 files
with h5py.File(common.fed_emnist_train, 'r+') as emnist_trn_h5:
    emnist_ex = emnist_trn_h5['examples']
    all_vals = []
    all_nps_trn = {}
    for val in emnist_ex:
        all_vals.append(val)
        all_nps_trn[val] = emnist_ex[val]['pixels'].value, emnist_ex[val]['label'].value

with h5py.File(common.fed_emnist_test, 'r') as emnist_tst_h5:
    emnist_ex = emnist_tst_h5['examples']
    all_nps_tst = {}
    for val in emnist_ex:
        all_nps_tst[val] = emnist_ex[val]['pixels'].value, emnist_ex[val]['label'].value

# Split the data into x,y pairs and save them as numpy arrays
print(len(all_nps_trn), len(all_nps_tst))
trn_size = sum([all_nps_trn[val][0].shape[0] for val in all_nps_trn])
tst_size = sum([all_nps_tst[val][0].shape[0] for val in all_nps_tst])
print(trn_size, tst_size)

trn_x = np.concatenate([all_nps_trn[val][0] for val in all_nps_trn])[:, :, :, None]
trn_y = np.concatenate([all_nps_trn[val][1] for val in all_nps_trn])

tst_x = np.concatenate([all_nps_tst[val][0] for val in all_nps_trn])[:, :, :, None]
tst_y = np.concatenate([all_nps_tst[val][1] for val in all_nps_trn])

print(trn_x.shape, trn_y.shape, tst_x.shape, tst_y.shape)

# Save the numpy arrays
np.save(common.emnist_trn_x_pth, trn_x)
np.save(common.emnist_trn_y_pth, trn_y)
np.save(common.emnist_tst_x_pth, tst_x)
np.save(common.emnist_tst_y_pth, tst_y)
