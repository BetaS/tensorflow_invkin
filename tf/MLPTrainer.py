import tensorflow as tf
from tf.util.tfutil import mlp
import numpy as np

tf.set_random_seed(0)


class MLPTrainer:
    def __init__(self, input_set, output_set, train_split=0.8, hidden_width=7, depth=3, learning_rate=0.01):
        self.train_size = int(len(input_set) * train_split)

        self.train_set = {"input": np.matrix(input_set[:self.train_size]), "output": np.matrix(output_set[:self.train_size])}
        self.test_set = {"input": np.matrix(input_set[self.train_size:]), "output": np.matrix(output_set[self.train_size:])}

        print("# train set: %d, test set : %d" % (len(self.train_set["input"]), len(self.test_set["input"])))

        self.input_width = len(input_set[0])
        self.output_width = len(output_set[0])
        self.hidden_width = hidden_width

        print("# input: %d nodes, output : %d nodes" % (self.input_width, self.output_width))

        # create placeholder
        self.input_holder = tf.placeholder(tf.float32, [None, self.input_width], name="X")
        self.output_holder = tf.placeholder(tf.float32, [None, self.output_width], name="Y")

        self.pred, self.weight, self.bias = mlp(self.input_holder, self.input_width, self.hidden_width, self.output_width, depth)

        # Define loss and optimizer
        #self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=y))
        #self.cost = tf.reduce_mean(self.pred-self.output_holder)
        self.cost = tf.nn.l2_loss(self.pred - self.output_holder)

        #self.optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        #self.optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.cost)
        self.optm = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        #self.corr = tf.equal(tf.argmax(self.pred, 1), tf.argmax(y, 1))
        #self.accr = tf.reduce_mean(tf.cast(self.corr, "float"))

        # Initializing the variables
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def training(self, training_epochs=200, display_epochs=1, batch_size=100):
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(self.train_size / batch_size)

            # Loop over all batches

            for start, end in zip(range(0, self.train_size, batch_size), range(batch_size, self.train_size, batch_size)):
                self.sess.run(self.optm, feed_dict={self.input_holder: self.train_set["input"][start:end], self.output_holder: self.train_set["output"][start:end]})
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
                cost = self.sess.run(self.cost, feed_dict={self.input_holder: self.train_set["input"], self.output_holder: self.train_set["output"]})
                accr = self.sess.run(self.cost, feed_dict={self.input_holder: self.test_set["input"], self.output_holder: self.test_set["output"]})
                print("Epoch: %03d/%03d cost: %.9f, accr : %.9f" % (epoch, training_epochs, cost, accr))

                #print(" W: ", self.sess.run(self.weight), ", b:", self.sess.run(self.bias))
                """
                train_acc = self.sess.run(self.cost, feed_dict={self.input_holder: self.train_set["input"], self.output_holder: self.train_set["output"]})
                print(" Training accuracy: %.3f" % (train_acc))

                test_acc = self.sess.run(self.cost, feed_dict={self.input_holder: self.test_set["input"], self.output_holder: self.test_set["output"]})
                print(" Test accuracy: %.3f" % (test_acc))
                """
                p = self.sess.run(self.pred, feed_dict={self.input_holder: self.test_set["input"][:5]})
                p2 = self.test_set["output"][:5]
                print(p.T)
                print(p2.T)

    def finish(self):
        self.sess.close()
