import numpy as np
import tensorflow as tf
from tensorflow import zeros_initializer as zinit
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.contrib.layers import xavier_initializer as xinit
from math import floor
from matplotlib import pyplot as plt

class RNN(object):
    def __init__( self, n_x=28, n_h=50, n_y=10, batch_size=100):
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y
        self.batch_size = batch_size

        self.create_weights()
        self.create_layers()
        self.create_gradients()

        # Init variables
        self.sess = tf.Session()
        self.sess.run( tf.global_variables_initializer())

        # Logger and TB sumaries

    def create_weights(self):
        # Placeholders
        self.x = tf.placeholder( dtype=tf.float32,
            shape=[None, self.n_x], name="x")
        self.y = tf.placeholder( dtype=tf.float32,
            shape=[None, self.n_y], name="y")
        # Previous timestep hidden layer
        # self.a_prev = tf.placeholder( shape=[None, self.n_h], dtype=tf.float32,
        #     name="a_prev")

        with tf.variable_scope( "weights"):

            # Input to Hidden
            self.w_input = tf.get_variable( shape=[self.n_x, self.n_h],
                initializer=xinit(), name="w_input") # I
            # Input bias incorporated in Hidden Bias

            # Previous Hidden to Current Hidden
            self.w_hidden = tf.get_variable( shape=[self.n_h, self.n_h],
                initializer=xinit(), name="w_hidden") # W
            self.b_hidden = tf.get_variable( shape=[self.n_h], initializer=zinit(),
                name="b_hidden") #Ba

            # Hidden to Output
            self.w_output = tf.get_variable( shape=[self.n_h, self.n_y],
                initializer=xinit(), name="w_output") # K
            self.b_output = tf.get_variable( shape=[self.n_y], initializer=zinit(),
                name="b_output") #Bb

    def create_layers(self):
        with tf.variable_scope("layers"):
            # Init a_prev layer
            self.a_prev = tf.get_variable( shape=[self.batch_size, self.n_h],
                initializer=zinit(), name="a_prev")
            # Current timestep unactivated
            self.a = tf.add(
                tf.add(
                    tf.matmul( self.x, self.w_input),
                    tf.matmul( self.a_prev, self.w_hidden)
                ),
                self.b_hidden, name="a"
            )
            # Current hidden activated with sigmoid
            self.h = tf.sigmoid( self.a, name="h")
            # Raw output layer
            self.b = tf.add( tf.matmul( self.h, self.w_output), self.b_output,
                name="b")
            self.o = tf.sigmoid( self.b, name="o")

        # Create History holders
        self.a_history = []


        with tf.variable_scope( "temporary_vars"):
            self.last_b = tf.get_variable( shape=[self.batch_size, self.n_y])
            self.last_h = tf.get_variable( shape=[self.batch_size, self.n_h])
            self.last_a = tf.get_variable( shape=[self.batch_size, self.n_h])

            self.tmp_last_a_grad = tf.get_variable(shape=[self.batch_size,
                self.n_h])


        # Create Gradient Holder
        self.d_last_prediction = tf.placeholder( dtype=tf.float32,
            shape=[self.batch_size, self.n_y])

    def fwd_pass(self, x_data, y_data):
        # print( x_data.shape)
        # input()
        # For the N-1 layers, compute the a component only (?)
        # Reset thing to be reset
        self.a_history = []
        tf.assign( self.a_prev, np.zeros([self.batch_size, self.n_h]))

        for slice_idx in range( n_x-1):
        # for x_slice in x_data[0:(n_x-1)]:
            # print( "Loop Processing tstep %d - Data: Skipped" % tstep)
            a = self.sess.run( self.a, feed_dict={ self.x:x_data[:,slice_idx,]})
            tf.assign( self.a_prev, a)
            self.a_history.append( a)
            print( "Current slice_idx %d" % slice_idx)

        a, self.last_h, self.last_b = self.sess.run( (self.a, self.h, self.b),
            feed_dict={ self.x: x_data[:,-1,]})
        self.a_history.append( a)
        tf.assign( self.last_a, a)

        prediction = self.sess.run( self.o,
            feed_dict={ self.x: x_data[:,-1,]})
        loss = self.sess.run( tf.reduce_sum(
            tf.square( tf.subtract( prediction, y_data))))
        # Premempitve computation
        self.d_last_prediction = self.sess.run(
            tf.subtract( prediction, y_data)) # delta L / delta o_N

        return prediction, loss

    def create_gradients(self):
        def sigmoid_prime( x):
            return tf.multiply( tf.sigmoid(x),
                tf.subtract( tf.constant(1.0), tf.sigmoid(x)))

        # Last timestep
        # delta L / delta b_N
        self.d_last_b = tf.multiply( self.d_last_prediction,
            sigmoid_prime( self.last_b))

        # delta L / delta h_N
        self.d_last_h = tf.matmul( self.d_last_b, tf.transpose( self.w_output))

        # delta L / delta K (w_output)
        self.d_last_w_output = tf.matmul( tf.transpose( self.last_h),
            self.d_last_h)

        # delta L / delta a_N
        self.d_last_a = tf.multiply( self.d_last_h,
            sigmoid_prime( self.last_a))

        # delta L / delta I (w_output)
        self.d_last_w_input = tf.matmul( tf.transpose( self.x), self.d_last_a)

        # for tstep in range( (self.n_x-1), 0, -1):
        # delta L / delta W_t
        self.d_w_hidden_t = tf.matmul( tf.transpose( self.tmp_last_a),
            self.tmp_a_grad)

        # delta L / delta a_previous
        self.d_a_prev = tf.matmul( self.tmp_a_grad, self.w_hidden)

        # delta L / delta w_input_prev
        self.d_w_input_prev = tf.matmul( tf.transpose(self.x),
            self.tmp_a_grad)

    def backprop(self, lr=.1, batch_x):

        # Holders for gradient results
        self.a_grads, self.w_hidden_grads, self.w_input_grads, \
            self.B_hidden_grads, self.B_output_grads = [], [], [], [], []

        feed_dict = { self.x: batch_x[:,(self.n_x-1),]}
        # Compute delta L / delta a_N : the last a layer gradient and saves
        self.a_grads.append( self.sess.run( self.d_last_a, feed_dict=feed_dict))

        # Compute delta L / delta B_hidden: same as the previous element
        self.B_hidden_grads.append( self.a_grads[-1])

        # Compute delta L / delta b_N
        d_last_b = self.sess.run( self.d_last_b, feed_dict=feed_dict)

        # Computes delta L / delta B_hidden
        self.B_output_grads.append( self.d_last_b)

        # Computes delta L / delta I (w_output)
        d_last_w_input = self.sess.run( self.d_last_w_input, feed_dict=feed_dict)
        self.w_input_grads.append( self.d_last_w_input)

        for tstep in range( (self.n_x-1), 0, -1):
            feed_dict = {self.x: batch_x[:,tstep,]}

            # Assign lastest tmp_a_grad and tmp_last_a
            tf.assing( self.)
            self.w_hidden_grads.append( self.sess.run( self.d_w_hidden_t,
                feed_dict=feed_dict))

            self.a_grads.append( self.sess.run( self.d_a_prev,
                feed_dict=feed_dict))

            self.B_hidden_grads.append( d_a_prev)

            self.w_input_grads.append( d_w_input_prev)

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
    batch_count=floor(len( x_train) / batch_size)

    for batch_idx in range(batch_count):
        batch_start=(batch_idx*batch_size)
        batch_x = x_train[batch_start:(batch_start+batch_size)]
        batch_y = y_train[batch_start:(batch_start+batch_size)]

        print( "Processing Batch [%d;%d]" % ( batch_start, batch_start+batch_size))

        prediction, loss = rnn.fwd_pass( batch_x, batch_y)
        # print( len(rnn.a_history[0][0]))
        # input()
        # d_last_b = rnn.sess.run( rnn.d_last_b ,
        #     feed_dict={ x: batch_x[-1]})
        # print( "Loss: ", loss)
