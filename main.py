import numpy as np
import tensorflow as tf
import tf.zeros_initializer as zinit
from tf.contrib.layers import xavier_initializer as xavinit
from tf.nn import softplus
from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt

# Init the seeds
np.random.seed(0)
tf.set_random_seed(0)

# Loading data
(x_train, y_train), (x_test, y_test) = load_data()

# print( x_train[0].shape)
# blacknwhite = np.array([[ (0 if x_train[0][i][j] / 255 < .5 else 1)
    # for j in range(28)] for i in range(28)])
# print(blacknwhite.shape)
# MNIST Format 28 rows, 28 cols, each col contains from 0 to 255 ?
# plt.imshow([ (0 if (x_train[0][j] / 255. < 0.5) else 1) for
    # j in range( len(x_train[0]))])
# plt.subplot(5, 2, 2*0 + 1)
# plt.imshow( x_train[0], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.subplot(5, 2, 2*1 + 1)
# plt.imshow( x_train[1], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
# plt.tight_layout()
# plt.show()

# print( y_train.shape)

class RNN(object):
    def __init__(self, nn_arch):
        self.session = tf.Session()

        self.x = tf.placeholder( shape=[None, arch_], name="x")
        self.y = tf.placeholder( shape=[None,nn_arch["n_output"]], name="y")

        # First Layer: Take X as input
        with tf.variable_scope( "l0"):
            self.w0 = tf.get_variable( name="w0",
                shape=[nn_arch["n_x"], nn_arch["n_input"]],
                initializer=xavinit())
            self.b0 = tf.get_variable( name="b0",
                shape=[nn_arch["n_input"], initializer=zinit())

            self.l0 = tf.add( tf.matmul(self.x, self.w0), self.b0, name="l0")

        # First hidden layer
        ## TODO: Take in account the previous timestep
        with tf.variable_scope( "h0"):
            self.wh0 = tf.get_variable( name="wh0", initializer=xinit(),
                shape=[nn_arch["n_input"],nn_arch["n_hidden_0"]])
            self.bh0 = tf.get_variable( name="bh0", initializer=zinit(),
                shape=[nn_arch["n_hidden_0"]])

            self.h0 = tf.add( tf.matmul( self.l0, self.wh0), self.bh0,
                name="h0")
            self.h0_prev = self.h0

        with tf.variable_scope( "o0"):
            self.wo0 = tf.get_variable( name="wo0", initializer=xinit(),
                shape=[nn_arch["n_output"]])
            self.bo0 = tf.get_variable( name="bo0", initializer=xinit(),
                shape=[nn_arch["n_output"]])

            self.o0 = tf.add( tf.matmul( self.h0, self.wo0), self.bo0,
                shape=[nn_arch["n_output"]], activation=tf.nn.softmax,
                name="o0" )

    def _initialize_weights(self, nn_arch):
        pass

if __name__ == "__main__":
    print( "So far so good")
    nn_arch = {
        "n_size" = 28,
        "n_input" = 100,
        "n_hidden_0" = 50,
        "n_output" = 10,
    }
