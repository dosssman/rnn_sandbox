import numpy as np
import tensorflow as tf
from tensorflow import zeros_initializer as zinit
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.contrib.layers import xavier_initializer as xinit

class RNN(object):
    def __init__( self, n_input=24, n_hidden=50, n_out=10):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.create_weights()
        self.create_layers()

        # Init variables
        self.sess = tf.Session()
        self.sess.run( tf.global_variables_initializer())

        # Logger and TB sumaries

    def create_weights(self):
        with tf.variable_scope( "weights"):
            # Placeholders
            self.x = tf.placeholder( dtype=tf.float32,
                shape=[None, self.n_input], name="x")
            self.y = tf.placeholder( dtype=tf.float32,
                shape=[None, self.n_out], name="y")

            # Input to Hidden
            self.w_input = tf.get_variable( shape=[self.n_input, self.n_hidden],
                initializer=xinit(), name="w_input")
            self.b_input = tf.get_variable( shape=[self.n_hidden], initializer=zinit(),
                name="b_input")

            # Previous Hidden to Current Hidden
            self.w_hidden = tf.get_variable( shape=[self.n_hidden, self.n_hidden],
                initializer=xinit(), name="w_hidden")
            self.b_hidden = tf.get_variable( shape=[self.n_hidden], initializer=zinit(),
                name="b_hidden")

            # Hidden to Output
            self.w_output = tf.get_variable( shape=[self.n_hidden, self.n_out],
                initializer=xinit(), name="w_output")
            self.b_output = tf.get_variable( shape=[self.n_out], initializer=zinit(),
                name="b_output")

    def create_layers(self):
        with tf.variable_scope("layers"):
            # Previous timestep for t = 0
            self.a_prev = tf.add( tf.matmul(
                tf.constant(0.0, shape=[ 1, self.n_input]), self.w_input),
                self.b_input, name="a_prev")

            # Input to hidden Layer
            self.input_to_hidden = tf.add( tf.matmul( self.x, self.w_input),
                self.b_input)
            self.prevhidden_to_hidden = tf.add(
                tf.matmul( self.a_prev, self.w_hidden),
                    self.w_hidden)

            self.a = tf.add( self.input_to_hidden, self.prevhidden_to_hidden,
                name="a")

            self.b = tf.sigmoid( self.a, name="b")

            self.raw_output = tf.add( tf.matmul( self.b, self.w_output), self.b_output,
                name="raw_output")

            self.prediction = tf.sigmoid( self.raw_output, name="prediction")

    def create_gradients(self):
        pass

    def step( self, x):
        pass

    def backprop(self, lr=.1):
        pass

    def train( self, x_train, y_train, n_epoch=1000, batch_size=100):
        for epoch in range( n_epoch):
            print( "Epoch %d" % epoch)
            batch_count = int( len(x_train) / batch_size)

            batch_start = 0
            for batch_idx in range( batch_count):
                batch_x = x_train[batch_start:(batch_start+batch_size)]
                batch_y = y_train[batch_start:(batch_start+batch_size)]

                # Process N-1 rows
                # for row in
                #     self.sess.run( self.prediction, feed_dict={ self.x: })

                batch_start+= batch_size

if __name__ == "__main__":
    # Dimensions
    n_input = 24
    n_hidden = 50
    n_out = 10

    # Loading data
    (x_train, y_train), (x_test, y_test) = load_data()

    y_train = np.array( [ np.array( [1 if y_train[i] == j else 0 for j in range(n_out)])
        for i in range( len(y_train))])
    y_test = np.array( [ np.array( [1 if y_test[i] == j else 0 for j in range(n_out)])
        for i in range( len(y_test))])

    rnn = RNN()
