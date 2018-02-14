import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import os
from training_plotter import TrainingPlotter
import logging.config
from utils import Utils

logging.config.fileConfig('logging.conf')


class Lenet5(object):
    def __init__(self, mnist_dataset, model_name, show_plot_window=False,
                 epochs=100, batch_size=500, variable_mean=0.,
                 variable_stddev=1., learning_rate=0.001, drop_out_keep_prob=0.5):
        self.file_name = os.getcwd() + '/results/Lenet5_{}_{}.png'.format(model_name, Utils.now_as_str())
        self.file_name_model = os.getcwd() + '/results/Lenet5_{}_{}.model.ckpt'.format(model_name,
                                                                                       Utils.now_as_str())
        self.file_name_confusion_matrix = os.getcwd() + '/results/Lenet5_confusion_matrix_{}_{}.png' \
            .format(model_name, Utils.now_as_str())
        self.file_name_wrong_predicts = os.getcwd() + '/results/Lenet5_wrong_predicts_{}_{}.png' \
            .format(model_name, Utils.now_as_str())
        title = "{}_{}_epochs_{}_batch_size_{}_learning_rate_{}_keep_prob_{}_variable_stddev_{}" \
            .format(self.__class__.__name__, model_name, epochs, batch_size,
                    learning_rate, drop_out_keep_prob, variable_stddev)
        self.plotter = TrainingPlotter(title,
                                       self.file_name,
                                       show_plot_window=show_plot_window)

        self.mnist_dataset = mnist_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.label_size = mnist_dataset.num_classes
        self.labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.variable_mean = variable_mean
        self.variable_stddev = variable_stddev

        logging.info(mnist_dataset.summary)

        self.session = None

        # clear the default graph
        tf.reset_default_graph()

        # consists of 32x32xcolor_channel
        color_channel = mnist_dataset.train.images.shape[3]
        self.x = tf.placeholder(tf.float32, (None, mnist_dataset.image_size, mnist_dataset.image_size, color_channel))

        self.y = tf.placeholder(tf.float32, (None, self.label_size))
        self.keep_prob = tf.placeholder(tf.float32)
        self.drop_out_keep_prob = drop_out_keep_prob
        self.network = Lenet5._LeNet(self, self.x, color_channel, variable_mean, variable_stddev)

        self.prediction_softmax = tf.nn.softmax(self.network)
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.network))
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.opt.minimize(self.loss_op)
        self.correct_prediction = tf.equal(tf.argmax(self.network, 1), tf.argmax(self.y, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    # create the LeNet architecture and return the result of the last fully connected layer.
    def _LeNet(self, x, color_channel, variable_mean, variable_stddev):

        # Hyperparameters
        patch_size = 5
        conv_layer_1_depth = 6
        conv_layer_2_depth = 16
        fc_layer_1_size = 400
        fc_layer_2_size = 120
        fc_layer_3_size = 84
        mu = variable_mean
        sigma = variable_stddev

        c1_weights = tf.Variable(
            tf.truncated_normal(shape=(patch_size, patch_size, color_channel, conv_layer_1_depth), mean=mu,
                                stddev=sigma))
        c1_biases = tf.Variable(tf.zeros(conv_layer_1_depth))
        c1 = tf.nn.conv2d(x, c1_weights, strides=[1, 1, 1, 1], padding='SAME') + c1_biases
        c1 = tf.nn.relu(c1)

        s2 = tf.nn.max_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # s2 = tf.nn.dropout(s2, self.keep_prob, seed=self.mnist_dataset.train.seed)

        c3_weights = tf.Variable(
            tf.truncated_normal(shape=(patch_size, patch_size, conv_layer_1_depth, conv_layer_2_depth), mean=mu,
                                stddev=sigma))
        c3_biases = tf.Variable(tf.zeros(conv_layer_2_depth))
        c3 = tf.nn.conv2d(s2, c3_weights, strides=[1, 1, 1, 1], padding='VALID') + c3_biases
        c3 = tf.nn.relu(c3)

        s4 = tf.nn.max_pool(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # s4 = tf.nn.dropout(s4, self.keep_prob, seed=self.mnist_dataset.train.seed)
        s4_flatten = flatten(s4)

        f5_weights = tf.Variable(tf.truncated_normal(shape=(fc_layer_1_size, fc_layer_2_size), mean=mu, stddev=sigma))
        f5_biases = tf.Variable(tf.zeros(fc_layer_2_size))
        f5 = tf.matmul(s4_flatten, f5_weights) + f5_biases
        f5 = tf.nn.relu(f5)
        f5 = tf.nn.dropout(f5, self.keep_prob, seed=self.mnist_dataset.train.seed)

        f6_weights = tf.Variable(tf.truncated_normal(shape=(fc_layer_2_size, fc_layer_3_size), mean=mu, stddev=sigma))
        f6_biases = tf.Variable(tf.zeros(fc_layer_3_size))
        f6 = tf.matmul(f5, f6_weights) + f6_biases
        f6 = tf.nn.relu(f6)
        f6 = tf.nn.dropout(f6, self.keep_prob, seed=self.mnist_dataset.train.seed)

        output_weights = tf.Variable(
            tf.truncated_normal(shape=(fc_layer_3_size, self.label_size), mean=mu, stddev=sigma))
        output_biases = tf.Variable(tf.zeros(self.label_size))
        logits = tf.matmul(f6, output_weights) + output_biases

        return logits

    def eval_data(self, dataset):
        steps_per_epoch = dataset.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        total_acc, total_loss = 0, 0
        sess = self.session
        # tf.get_default_session()
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(self.batch_size)
            loss, acc = sess.run([self.loss_op, self.accuracy_op], feed_dict={self.x: batch_x, self.y: batch_y,
                                                                              self.keep_prob: 1.0})
            total_acc += (acc * batch_x.shape[0])
            total_loss += (loss * batch_x.shape[0])
        return total_loss / num_examples, total_acc / num_examples

    def test_data(self, dataset):
        steps_per_epoch = dataset.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        total_acc, total_loss = 0, 0
        total_predict, total_actual = [], []
        wrong_predict_images = []

        # tf.get_default_session()
        sess = self.session
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(self.batch_size)
            loss, acc, predict, actual = sess.run(
                [self.loss_op, self.accuracy_op, tf.argmax(self.network, 1), tf.argmax(self.y, 1)],
                feed_dict={self.x: batch_x, self.y: batch_y,
                           self.keep_prob: 1.0})
            total_acc += (acc * batch_x.shape[0])
            total_loss += (loss * batch_x.shape[0])
            total_predict = np.append(total_predict, predict)
            total_actual = np.append(total_actual, actual)
            for index in range(len(predict)):
                if predict[index] != actual[index]:
                    wrong_predict_images.append(batch_x[index])

        return total_loss / num_examples, total_acc / num_examples, total_predict, total_actual, wrong_predict_images

    def train(self):
        saver = tf.train.Saver()
        if self.session is not None:
            self.session.close()
        with tf.Session() as self.session:
            self.session.run(tf.initialize_all_variables())
            steps_per_epoch = self.mnist_dataset.train.num_examples // self.batch_size
            num_examples = steps_per_epoch * self.batch_size
            # Train model
            for i in range(self.epochs):
                self.mnist_dataset.train.shuffle()
                total_tran_loss = 0.0
                total_tran_acc = 0.0
                for step in range(steps_per_epoch):
                    batch_x, batch_y = self.mnist_dataset.train.next_batch(self.batch_size)
                    _, train_loss, train_acc = self.session.run(
                        [self.train_op, self.loss_op, self.accuracy_op],
                        feed_dict={self.x: batch_x, self.y: batch_y, self.keep_prob: self.drop_out_keep_prob})
                    total_tran_loss += (train_loss * batch_x.shape[0])
                    total_tran_acc += (train_acc * batch_x.shape[0])

                total_tran_loss = total_tran_loss / num_examples
                total_tran_acc = total_tran_acc / num_examples
                val_loss, val_acc = self.eval_data(self.mnist_dataset.validation)
                logging.info(
                    "EPOCH {} --- Training: loss = {:.3f}, accuracy = {:.3f}; Validation: loss = {:.3f}, accuracy = {:.3f};"
                        .format(i + 1, total_tran_loss, total_tran_acc, val_loss, val_acc))
                self.plotter.add_loss_accuracy_to_plot(i, total_tran_loss, total_tran_acc, val_loss, val_acc,
                                                       redraw=True)

            saver.save(self.session, self.file_name_model)
            logging.info("Model saved into {}".format(self.file_name_model))

            # Evaluate on the test data
            test_loss, test_acc, total_predict, total_actual, wrong_predict_images = self.test_data(
                self.mnist_dataset.test)
            logging.info("Test loss = {:.3f} accuracy = {:.3f}".format(test_loss, test_acc))
            self.plotter.plot_confusion_matrix(
                total_actual, total_predict, self.labels_name).savefig(self.file_name_confusion_matrix)
            try:
                # before plotting, sort images by true target label
                wrong_actual = total_actual[total_actual != total_predict]
                wrong_predict_images = np.array(wrong_predict_images)
                wrong_predict_images_sorted = wrong_predict_images[wrong_actual.argsort(),]
                wrong_predict_images_sorted = [image for image in wrong_predict_images_sorted]
                self.plotter.combine_images(wrong_predict_images_sorted, self.file_name_wrong_predicts)
            except Exception as ex:
                print("Failed when plotting wrong predicted images:\n" + str(ex))
        self.plotter.safe_shut_down()

    def predict_images(self, images):
        return self.session.run(self.prediction_softmax, feed_dict={self.x: images, self.keep_prob: 1.0})

    def restore_session(self, ckpt_dir, ckpt_filename=None):
        """
        Function for restore a model session from previous saved ones

        :param ckpt_dir: a directory for checkpoint to search in
        :param ckpt_filename: try to restore model with this checkpoint file name; if is None, restore a model using
                              latest_checkpoint method from ckpt_dir directory
        """

        dir = os.path.dirname(ckpt_dir)

        # check if directory  ckpt_dir exists
        if not os.path.exists(dir):
            print('Directory {} not found.'.format(ckpt_dir))
        else:
            if self.session is not None:
                self.session.close()
            self.session = tf.Session()
            saver = tf.train.Saver()
            if ckpt_filename is not None:
                saver.restore(sess=self.session, save_path=os.path.join(dir, ckpt_filename))
            else:
                saver.restore(sess=self.session, save_path=tf.train.latest_checkpoint(ckpt_dir))
