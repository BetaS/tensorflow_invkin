import tensorflow as tf
from tf.util.tfutil import mlp
import numpy as np

tf.set_random_seed(0)


class MLPTrainer:
    def __init__(self, input_set, output_set, hidden_width=7, depth=3, learning_rate=0.01):
        self.train_size = int(len(input_set) * 0.9)

        self.train_set = {"input": input_set[:self.train_size], "output": output_set[:self.train_size]}
        self.test_set = {"input": input_set[self.train_size:], "output": output_set[self.train_size:]}

        self.input_width = len(input_set[0])
        self.output_width = len(output_set[0])
        self.hidden_width = hidden_width

        # create placeholder
        x = tf.placeholder("float", [None, self.input_width])
        y = tf.placeholder("float", [None, self.output_width])

        self.input_holder = x
        self.output_holder = y

        self.pred = mlp(x, self.input_width, self.hidden_width, self.output_width, depth)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=y))
        self.optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.corr = tf.equal(tf.argmax(self.pred, 1), tf.argmax(y, 1))
        self.accr = tf.reduce_mean(tf.cast(self.corr, "float"))

        # Initializing the variables
        self.init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init)

    def training(self, training_epochs=200, display_epochs=10, batch_size=100):
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(self.train_size / batch_size)

            # Loop over all batches
            for i in range(total_batch):
                randidx = np.random.randint(self.train_size, size=batch_size)
                print(randidx)

                # picking batch set
                batch_xs = self.train_set["input"][randidx, :]
                batch_ys = self.train_set["output"][randidx, :]

                # Fit training using batch data
                self.sess.run(self.optm, feed_dict={self.input_holder: batch_xs, self.output_holder: batch_ys})

                # Compute average loss
                avg_cost += self.sess.run(self.cost, feed_dict={self.input_holder: batch_xs, self.output_holder: batch_ys}) / total_batch


            # Display logs per epoch step
            if epoch % display_epochs == 0:
                print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))

                train_acc = self.sess.run(self.accr, feed_dict={self.input_holder: self.train_set["input"], self.output_holder: self.train_set["output"]})
                print(" Training accuracy: %.3f" % (train_acc))

                test_acc = self.sess.run(self.accr, feed_dict={self.input_holder: self.test_set["input"], self.output_holder: self.test_set["output"]})
                print(" Test accuracy: %.3f" % (test_acc))

    def finish(self):
        self.sess.close()
