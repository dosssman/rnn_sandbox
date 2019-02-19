import time
import numpy as np
import tensorflow as tf
from tensorflow import zeros_initializer as zinit
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.contrib.layers import xavier_initializer as xinit
# import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = load_data()

n_x = 784
n_z = 72
n_y = 10

x_train = np.reshape( x_train, [len(x_train), 784 ])
x_test = np.reshape( x_test, [len(x_test), 784 ])

#Reshaping to one hots
y_train = np.array( [ np.array( [1 if y_train[i] == j else 0 for j in range(n_y)])
    for i in range( len(y_train))])
y_test = np.array( [ np.array( [1 if y_test[i] == j else 0 for j in range(n_y)])
    for i in range( len(y_test))])

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

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

        variable_summaries( self.w0)
        variable_summaries( self.b0)
        variable_summaries( self.w_out)
        variable_summaries( self.b_out)

        # Init
        self.create_layers()
        self.create_backprop()
        self.create_gradient_step()
        self.create_eval()

        # TF init
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run( init)

        self.cost = tf.reduce_sum( tf.square(
            tf.subtract( self.output, self.y)))

        tf.summary.scalar( "Loss", self.cost)
        self.merged_summary = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter( "./logs/" +
            time.strftime("%Y-%m-%d-%H-%M-%S"), self.sess.graph)

    def create_layers( self):
        with tf.variable_scope( "layers"):
            self.z0 = tf.add( tf.matmul( self.x, self.w0), self.b0, name="z0")
            self.l0 = tf.sigmoid( self.z0, name="l0")

            self.z_output = tf.add( tf.matmul( self.l0, self.w_out), self.b_out,
                name="z_output")
            self.output = tf.sigmoid( self.z_output, name="output")

            tf.summary.histogram( "z0", self.z0)
            tf.summary.histogram( "z_output", self.z_output)

            tf.summary.histogram( "l0", self.l0)
            tf.summary.histogram( "output", self.output)

        return self.output

    def create_backprop( self): #???
        with tf.variable_scope("deltas"):
            diff = tf.subtract( self.output, self.y)

            self.d_z_out = tf.multiply( diff,
                self.sigmoid_prime( self.z_output), name="d_z_out")
            self.d_b_out = self.d_z_out
            self.d_w_out = tf.matmul( tf.transpose( self.l0),
                self.d_z_out, name="d_w_out")

            self.d_l0 = tf.matmul( self.d_z_out, tf.transpose( self.w_out),
                name="d_l0")
            self.d_z0 = tf.multiply( self.d_l0, self.sigmoid_prime( self.z0),
                name="d_z0")
            self.d_b0 = self.z0
            self.d_w0 = tf.matmul( tf.transpose( self.x), self.d_z0, name="d_w0")

            tf.summary.histogram( "Diff", diff)
            tf.summary.histogram( "d_z_out__d_b_out", self.d_z_out)
            tf.summary.histogram( "d_w_out", self.d_w_out)
            tf.summary.histogram( "d_l0", self.d_l0)
            tf.summary.histogram( "d_z0__d_b0", self.d_z0)
            tf.summary.histogram( "d_w0", self.d_w0)


    def create_gradient_step( self, lr=tf.constant(0.5)):
        self.step = [
            tf.assign( self.w_out, tf.subtract( self.w_out,
                tf.multiply( lr, self.d_w_out)), name="w_out_grad_step"),
            tf.assign( self.b_out, tf.subtract( self.b_out,
                tf.multiply( lr, tf.reduce_mean( self.d_b_out, axis=[0]))),
                name="b_out_grad_step"),
            tf.assign( self.b0, tf.subtract( self.b0,
                tf.multiply( lr, tf.reduce_mean( self.d_b0, axis=[0]))),
                name="b0_grad_step"),
            tf.assign( self.w0, tf.subtract( self.w0,
                tf.multiply( lr, self.d_w0)), name="w0_grad_step")
        ]

    def create_eval(self):
        acct_mat = tf.equal( tf.argmax( self.output, 1),
            tf.argmax( self.y, 1))
        self.acct_res = tf.reduce_sum( tf.cast( acct_mat, tf.float32))

        # tf.summary.scalar( "Acc", self.acct_res)

        return self.acct_res

    def train( self, x_train, y_train, lr=.1, batch_size=1000, max_epoch=10000):
        with self.sess as sess:

            for epoch in range(max_epoch):
                batch_start = 0
                # np.random.shuffle( x_train)
                # np.random.shuffle( y_train)

                batch_count = int(len(x_train) / batch_size)

                for batch_idx in range( batch_count):
                    # print( "Processing batch %d: [%d:%d]" % (batch_idx,
                    #     batch_start, batch_start+batch_size))
                    batch_x, batch_y = \
                        x_train[batch_start:(batch_start+batch_size)], \
                        y_train[batch_start:(batch_start+batch_size)]

                    step, cost, merged_summary = sess.run( [self.step, self.cost, self.merged_summary],
                        feed_dict={ self.x: batch_x, self.y: batch_y})

                    self.train_writer.add_summary( merged_summary, (epoch*batch_count)+batch_idx)

                    res = sess.run( self.acct_res,
                        feed_dict={ self.x: x_test[:1000],
                                    self.y: y_test[:1000]})

                    # self.train_writer.add_summary( acct_res, (epoch*batch_count)+batch_idx)

                    batch_start += batch_size

                if epoch % 10 == 0 and epoch > 0:
                    print( "Result @Epoch %d: %.3f" % ( epoch, res))

    def sigmoid( self, x):
        return tf.div( tf.constant(1.0),
            tf.add( tf.constant( 1.0), tf.negative( x)))

    def sigmoid_prime( self, x):
        return tf.multiply( self.sigmoid(x),
            tf.subtract( tf.constant(1.0), x))

if __name__ == "__main__":

    nn = NN( n_x, n_z, n_y)
    nn.train( x_train, y_train)
