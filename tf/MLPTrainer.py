import tensorflow as tf
from tf.util.tfutil import mlp
import numpy as np
import time
import matplotlib.pyplot as plt


class MLPTrainer:
    def __init__(self, input_width, output_width, hidden_width=7, depth=3, learning_rate=0.01, file_name="checkpoint_0", from_file=False):
        self.save_path = "./save/"+file_name+".ckpt"

        self.train_size = 0
        self.train_set = {"input": [], "output": []}

        self.test_size = 0
        self.test_set = {"input": [], "output": []}

        self.input_width = input_width
        self.output_width = output_width
        self.hidden_width = hidden_width

        print("# input: %d nodes, output : %d nodes" % (self.input_width, self.output_width))

        # create placeholder
        self.input_holder = tf.placeholder(tf.float32, [None, self.input_width], name="X")
        self.output_holder = tf.placeholder(tf.float32, [None, self.output_width], name="Y")

        self.pred, self.weight, self.bias = mlp(self.input_holder, self.input_width, self.hidden_width, self.output_width, depth)

        # Define loss and optimizer
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.output_holder))
        #self.cost = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(self.pred-self.output_holder, 2), 1)))
        #self.cost = tf.losses.absolute_difference(self.pred, self.output_holder)
        #self.cost = tf.reduce_mean(tf.square(self.pred-self.output_holder))
        #self.cost = tf.nn.l2_loss(self.pred - self.output_holder)
        #self.cost = tf.reduce_mean(tf.squared_difference(self.pred, self.output_holder))
        self.cost = tf.reduce_mean(tf.abs(tf.tanh(self.pred-self.output_holder)))

        self.optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        #self.optm = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        self.corr = tf.losses.absolute_difference(self.pred, self.output_holder)
        #self.accr = tf.reduce_mean(tf.cast(self.corr, "float"))

        self.saver = tf.train.Saver()

        # Initializing the variables
        self.sess = tf.Session()

        if from_file:
            self.saver.restore(self.sess, self.save_path)
        else:
            self.sess.run(tf.initialize_all_variables())

        self.avgs = []
        self.accrs = []
        self.error_set = [[], [], [], [], [], [], []]

    def set_tester(self, input_set, output_set):
        self.test_size = len(input_set)
        self.test_set = {"input": np.matrix(input_set), "output": np.matrix(output_set)}

    def training(self, input_set, output_set, training_epochs=200, display_epochs=100, batch_size=100):
        self.train_size = len(input_set)
        self.train_set = {"input": np.matrix(input_set), "output": np.matrix(output_set)}

        # Training cycle
        for epoch in range(1, training_epochs+1):
            avg_cost = 0.
            total_batch = int(self.train_size / batch_size)

            # Loop over all batches

            for start, end in zip(range(0, self.train_size, batch_size), range(batch_size, self.train_size, batch_size)):
                c, _ = self.sess.run([self.cost, self.optm], feed_dict={self.input_holder: self.train_set["input"][start:end], self.output_holder: self.train_set["output"][start:end]})
                avg_cost += c / total_batch

            """
            for i in range(total_batch):
                randidx = np.random.randint(self.train_size, size=batch_size)

                # picking batch set
                batch_xs = self.train_set["input"][randidx, :]
                batch_ys = self.train_set["output"][randidx, :]

                # Fit training using batch data
                self.sess.run(self.optm, feed_dict={self.input_holder: batch_xs, self.output_holder: batch_ys})
    
            """

            # Display logs per epoch step
            if epoch % display_epochs == 0:
                accr = self.sess.run(self.corr, feed_dict={self.input_holder: self.test_set["input"], self.output_holder: self.test_set["output"]})# / self.test_size
                print("Epoch: %03d/%03d , avg: %.9f, accr : %.9f" % (epoch, training_epochs, avg_cost, accr))

                self.avgs.append(avg_cost)
                self.accrs.append(accr)

                #print(" W: ", self.sess.run(self.weight), ", b:", self.sess.run(self.bias))
                """
                train_acc = self.sess.run(self.cost, feed_dict={self.input_holder: self.train_set["input"], self.output_holder: self.train_set["output"]})
                print(" Training accuracy: %.3f" % (train_acc))

                test_acc = self.sess.run(self.cost, feed_dict={self.input_holder: self.test_set["input"], self.output_holder: self.test_set["output"]})
                print(" Test accuracy: %.3f" % (test_acc))
                """
                p = self.sess.run(self.pred, feed_dict={self.input_holder: self.test_set["input"][0]})
                p2 = self.test_set["output"][0]
                diff = (p2-p).T
                for i in range(len(diff)):
                    self.error_set[i].append(float(diff[i]))

        self.saver.save(self.sess, self.save_path)
        print("Checkpoint saved")


    def finish(self):
        self.sess.close()
