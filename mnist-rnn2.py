import numpy as np
import tensorflow as tf
from tensorflow import zeros_initializer as zinit
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.contrib.layers import xavier_initializer as xinit
from math import floor
from matplotlib import pyplot as plt


class RNN(object):
    def __init__(self, n_x=28, n_h=50, n_y=10, batch_size=100):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.batch_size = batch_size

        self.create_layers()

        # Init variables
        self.optimizer = tf.train.AdamOptimizer()

        self.sess = tf.Session()
        self.sess.run( tf.global_variables_initializer())

    def create_layers( self):
        # Placeholders
        self.x = tf.placeholder( dtype=tf.float32,
            shape=[None, self.n_x], name="x")
        self.y = tf.placeholder( dtype=tf.float32,
            shape=[None, self.n_y], name="y")

        with tf.variable_scope( "layers"):
            # Previous hidden layer
            sefl.previous_h0 = tf.get_variable(shape=[self.bacth_size,self.n_h],
                name="previous_h0", initializer=zinit())
            # Hidden Layer
            self.h0=tf.layers.dense( self.x, units=self.n_h,
                activation="sigmoid", kernel_initializer=xinit(), name="h0")
            self.h0 = 
            # Output Layer
            self.output = tf.layers.dense( self.h0, units=self.n_y,
                activation="softmax", name="output")

    def forward_pass( self, batch_x, batch_y):

        pass

    def backprop( self):
        pass

    def train(self):
        pass

if __name__ == "__main__":
    # Dimensions
    n_x = 28
    n_h = 50
    n_y = 10
    batch_size=100

    # Loading data
    (x_train, y_train), (x_test, y_test) = load_data()

    y_train = np.array( [ np.array( [1 if y_train[i] == j else 0 for j in range(n_y)])
        for i in range( len(y_train))])
    y_test = np.array( [ np.array( [1 if y_test[i] == j else 0 for j in range(n_y)])
        for i in range( len(y_test))])

    rnn = RNN(n_x=n_x, n_y=n_y, n_h=n_h, batch_size=batch_size)


    # TODO Move to train()
    # batch_count=floor(len( x_train) / batch_size)
    #
    # for batch_idx in range(batch_count):
    #     batch_start=(batch_idx*batch_size)
    #     batch_x = x_train[batch_start:(batch_start+batch_size)]
    #     batch_y = y_train[batch_start:(batch_start+batch_size)]
    #
    #     prediction, loss = rnn.fwd_pass( batch_x, batch_y)
    #     print( "Processeced Batch [%d;%d] - Loss: %.3f" % ( batch_start, batch_start+batch_size, loss))
    #
    #     rnn.backprop( batch_x)

        # print( len(rnn.a_history[0][0]))
        # input()
        # d_last_b = rnn.sess.run( rnn.d_last_b ,
        #     feed_dict={ x: batch_x[-1]})
        # print( "Loss: ", loss)
