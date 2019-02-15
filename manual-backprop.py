import numpy as np
import tensorflow as tf
from tensorflow import zeros_initializer as zinit
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.contrib.layers import xavier_initializer as xinit
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = load_data()

n_x = 784
n_z = 100
n_y = 10

x_train = np.reshape( x_train, [len(x_train), 784 ])
x_test = np.reshape( x_test, [len(x_test), 784 ])

#Reshaping to one hots
y_train = np.array( [ np.array( [1 if y_train[i] == j else 0 for j in range(n_y)])
    for i in range( len(y_train))])
y_test = np.array( [ np.array( [1 if y_test[i] == j else 0 for j in range(n_y)])
    for i in range( len(y_test))])

class NN(object):
    def __init__(self, n_x, n_h, n_y):
        # Vars and tensorts
        self.x = tf.placeholder( dtype=tf.float32, shape=[None, 784])
        self.y = tf.placeholder( dtype=tf.float32, shape=[None, 10])

        with tf.variable_scope( "weights"):
            self.w0 = tf.get_variable( shape=[n_x, n_h], initializer=xinit(),
                name="w0")
            self.b0 = tf.get_variable( shape=[n_h], initializer=zinit(),
                name="b0")
            self.w_out = tf.get_variable( shape=[n_h, n_y], initializer=xinit(),
                name="w_out")
            self.b_out = tf.get_variable( shape=[n_y], initializer=xinit(),
                name="b_out")

        # Init
        self.create_layers()
        self.create_backprop()
        self.create_gradient_step()

        # TF init
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run( init)

        # self.cost = tf.square( tf.subtract( selfoutput, self.y))

    def create_layers( self):
        with tf.variable_scope( "layers"):
            self.z0 = tf.add( tf.matmul( self.x, self.w0), self.b0, name="z0")
            self.l0 = tf.sigmoid( self.z0, name="l0")

            self.z_output = tf.add( tf.matmul( self.l0, self.w_out), self.b_out,
                name="z_output")
            self.output = tf.sigmoid( self.z_output, name="output")

        return self.output

    def create_backprop( self): #???
        with tf.variable_scope("deltas"):
            diff = tf.subtract( self.output, self.y)

            self.d_z_out = tf.multiply( diff,
                self.sigmoid_prime( self.z_output), name="d_z_out")
            self.d_b_out = self.d_z_out
            self.d_w_out = tf.matmul( tf.transpose( self.l0),
                self.d_z_out, name="d_w_out")

            self.d_l0 = tf.matmul( self.d_z_out, tf.transpose( self.w_out))
            self.d_z0 = tf.multiply( self.d_l0, self.sigmoid_prime( self.z0))
            self.d_b0 = self.z0
            self.d_w0 = tf.matmul( tf.transpose( self.x), self.d_z0)

    def create_gradient_step( self, lr=0.1):
        self.step = [
            tf.assign( self.w_out, tf.subtract( self.w_out,
                tf.multiply( lr, self.d_w_out))),
            tf.assign( self.b_out, tf.subtract( self.b_out,
                tf.multiply( lr, self.d_b_out))),
            tf.assign( self.b0, tf.subtract( self.b0,
                tf.multiply( lr, self.d_b0))),
            tf.assign( self.w0, tf.subtract( self.w0,
                tf.multiply( lr, self.d_w0)))
        ]
        # Session run etc ...

    def train( self, x_train, y_train, lr=.1, batch_size=100, max_epoch=10000):
        with self.sess as sess:
            acct_mat = tf.equal( tf.argmax( self.output, 1), tf.argmax( self.y, 1))
            acct_res = tf.reduce_sum( tf.cast( acct_mat, tf.float32))

            for epoch in range(max_epoch):
                batch_start = 0

                for _ in range(int(len(x_train) / batch_size)):
                    batch_x, batch_y = x_train[batch_start:(batch_start+batch_size)], \
                        y_train[batch_start:(batch_start+batch_size)]

                    sess.run( self.step, feed_dict={ self.x: batch_x, self.y: batch_y})
                    batch_start += batch_size

                    if epoch % 1000 == 0:
                        res = sess.run( acct_res,feed_dict={ self.x: x_test[:1000],
                            self.y: y_test[:1000]})

                        print(res)

    def sigmoid( self, x):
        return tf.div( tf.constant(1.0),
            tf.add( tf.constant( 1.0), tf.negative( x)))

    def sigmoid_prime( self, x):
        return tf.multiply( self.sigmoid(x),
            tf.subtract( tf.constant(1.0), x))

if __name__ == "__main__":

    nn = NN( n_x, n_z, n_y)
    nn.train( x_train, y_train)
