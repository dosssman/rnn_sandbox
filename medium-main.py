import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = load_data()

# print( x_train[0].shape)
# blacknwhite = np.array([[ (0 if x_train[0][i][j] / 255 < .5 else 1)
    # for j in range(28)] for i in range(28)])
# print(blacknwhite.shape)
# MNIST Format 28 rows, 28 cols, each col contains from 0 to 255 ?
# plt.imshow([ (0 if (x_train[0][j] / 255. < 0.5) else 1)
    # for j in range( len(x_train[0]))])
# plt.subplot(5, 2, 2*0 + 1)
# plt.imshow( x_train[0], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.subplot(5, 2, 2*1 + 1)
# plt.imshow( x_train[1], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.tight_layout()
# plt.show()
