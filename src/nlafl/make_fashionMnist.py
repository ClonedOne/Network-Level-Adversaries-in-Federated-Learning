"""
Create the Fashion MNIST dataset using tf.keras.datasets.fashion_mnist.load_data().
"""

import numpy as np
import tensorflow as tf
import common

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)
print("x_train.shape: ", x_train.shape)
print("x_test.shape: ", x_test.shape)
print("y_train.shape: ", y_train.shape)
print("y_test.shape: ", y_test.shape)


#Save the numpy arrays
np.save(common.fashionMnist_trn_x_pth, tf.cast(x_train, tf.float32) / 255.0)
np.save(common.fashionMnist_trn_y_pth, y_train)
np.save(common.fashionMnist_tst_x_pth, tf.cast(x_test, tf.float32) / 255.0)
np.save(common.fashionMnist_tst_y_pth, y_test)