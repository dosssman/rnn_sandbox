import numpy as np
import tensorflow as tf
from tensorflow import zeros_initializer as zinit
from tensorflow.contrib.layers import xavier_initializer as xinit

n_input = 24
n_hidden = 50
n_out = 50

class RNN(object):
    def __init__( self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.create_weights()

    def create_weights(self):
        with variable_scope( "weights"):
            # Input to Hidden
            self.w_input = tf.get_variable( shape=[n_input, n_hidden],
                initializer=xinit(), name="w_input")
            self.b_input = tf.get_variable( shape=[n_hidden], initializer=zinit(),
                name="b_input")

            # Previous Hidden to Current Hidden
            self.w_hidden = tf.get_variable( shape=[n_hidden, n_hidden],
                initializer=xinit(), name="w_hidden")
            self.b_hidden = tf.get_variable( shape=[n_hidden], initializer=zinit(),
                name="b_hidden")

            # Hidden to Output
            self.w_output = tf.get_variable( shape=[n_hidden, n_output],
                initializer=xinit(), name="w_output")
            self.b_output = tf.get_variable( shape=[n_output], initializer=zinit(),
                name="b_output")

    def create_layers(self):
        with variable_scope("preactivations"):
            # Input to hidden Layer
            self.input_to_hidden = tf.add( tf.matmul( self.x, self.w_input),
                self.b_input)
            self.prevhidden_to_hidden = tf.add(
                tf.matmul( self.a_prev, self.w_hidden),
                tf.w_hidden)

            self.a = tf.add( self.input_to_hidden, self.prevhidden_to_hidden,
                name="a")

        with variable_scope( "activations"):
            self.b = tf.sigmoid( )
