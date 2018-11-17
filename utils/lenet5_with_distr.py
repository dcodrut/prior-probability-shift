import logging.config
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from utils import utils
from utils.training_plotter import TrainingPlotter

logging.config.fileConfig('logging.conf')


class Lenet5WithDistr(object):
    """
        Implements a Lenet5 architecture and some methods for training, test and evaluation
        As input, along with the image, we add label distribution
    """

    def __init__(self, dataset, save_dir='results/', model_name='no_name', show_plot_window=False,
                 epochs=100, batch_size=500, variable_mean=0.,
                 variable_stddev=1., learning_rate=0.001, drop_out_keep_prob=0.5, verbose=True,
                 distr_pos=[False, False, False, False, False]):

        if distr_pos is None:
            distr_pos_to_save = -1
        else:
            distr_pos_to_save = distr_pos
        if distr_pos is None or (isinstance(distr_pos, np.int32) and distr_pos == -1):
            distr_pos = [False, False, False, False, False]

        self.verbose = verbose
        save_dir_full_path = os.path.join(os.getcwd(), os.path.dirname(save_dir))
        base_file_name = '{}/{}_{}_{}'.format(save_dir_full_path, self.__class__.__name__, model_name,
                                              utils.now_as_str())
        self.file_name_learning_curve = '{}.learning_curve.png'.format(base_file_name)
        self.file_name_model = '{}.model.ckpt'.format(base_file_name)
        self.file_name_confusion_matrix = '{}.confusion_matrix.png'.format(base_file_name)
        self.file_name_wrong_predicts = '{}.wrong_predicts.png'.format(base_file_name)

        self.plotter = TrainingPlotter(title="{}_{}".format(self.__class__.__name__, model_name),
                                       file_name=self.file_name_learning_curve, show_plot_window=show_plot_window)

        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.label_size = dataset.num_classes
        self.labels_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.variable_mean = variable_mean
        self.variable_stddev = variable_stddev

        if self.verbose:
            logging.info(dataset.summary)

        self.session = None

        # clear the default graph
        tf.reset_default_graph()

        color_channel = dataset.train.images.shape[3]
        self.x = tf.placeholder(tf.float32, (None, dataset.image_size, dataset.image_size, color_channel))
        self.y = tf.placeholder(tf.float32, (None, self.label_size))
        self.train_distr = tf.Variable(initial_value=self.dataset.train.label_distr, name='train_distr')
        self.test_distr = tf.Variable(initial_value=self.dataset.test.label_distr, name='test_distr')
        self.distr_pos = tf.Variable(initial_value=distr_pos_to_save, name='distr_pos')
        self.train_num_examples = tf.Variable(initial_value=self.dataset.train.num_examples,
                                              name='train_num_examples')
        self.y_distr = tf.placeholder(tf.float32, (self.label_size,), name='y_distr')  # the new input (label distr.)
        self.keep_prob = tf.placeholder(tf.float32)
        self.drop_out_keep_prob = drop_out_keep_prob
        self.network = Lenet5WithDistr._LeNet(self, self.x, color_channel, variable_mean, variable_stddev, distr_pos)

        self.prediction_softmax = tf.nn.softmax(self.network)
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.network))
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.opt.minimize(self.loss_op)
        self.correct_prediction = tf.equal(tf.argmax(self.network, 1), tf.argmax(self.y, 1))
        self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def _LeNet(self, x, color_channel, variable_mean, variable_stddev, distr_pos):
        """
        Implementation of the LeNet5 architecture plus label distribution as input
        :param x: input images
        :param color_channel:
        :param variable_mean:
        :param variable_stddev:
        :return: the result of the last fully connected layer
        """
        # Hyperparameters
        distr_size = self.dataset.num_classes
        patch_size = 5
        conv_layer_1_depth = 6
        conv_layer_2_depth = 16
        fc_layer_1_size = 400
        fc_layer_2_size = 120
        fc_layer_3_size = 84
        mu = variable_mean
        sigma = variable_stddev

        # tile y_distr in order to attach it as input for network's layers
        distr_to_concat_fc = tf.reshape(tf.tile(input=self.y_distr, multiples=[tf.shape(x)[0]]),
                                        shape=[tf.shape(x)[0], distr_size])
        self.batch_distr = distr_to_concat_fc
        # implement network architecture
        c1_weights = tf.Variable(
            tf.truncated_normal(shape=(patch_size, patch_size, color_channel, conv_layer_1_depth), mean=mu,
                                stddev=sigma))
        c1_biases = tf.Variable(tf.zeros(conv_layer_1_depth))
        if distr_pos[0]:
            if self.verbose:
                logging.debug('Attached distr. before C1 (at input)')
            temp = tf.concat([tf.zeros(x.shape[2] - distr_to_concat_fc.shape[1]), self.y_distr], axis=0)
            distr_to_concat_c = tf.reshape(tf.tile(input=temp, multiples=[tf.shape(x)[0]]),
                                           shape=[tf.shape(x)[0], 1, int(temp.shape[0]), 1])
            x = tf.concat([x, distr_to_concat_c], axis=1)
        c1 = tf.nn.conv2d(x, c1_weights, strides=[1, 1, 1, 1], padding='SAME') + c1_biases
        c1 = tf.nn.relu(c1)

        s1 = tf.nn.max_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # s1 = tf.nn.dropout(s1, self.keep_prob, seed=self.mnist_dataset.train.seed)

        c2_weights = tf.Variable(
            tf.truncated_normal(shape=(patch_size, patch_size, conv_layer_1_depth, conv_layer_2_depth), mean=mu,
                                stddev=sigma))
        c2_biases = tf.Variable(tf.zeros(conv_layer_2_depth))

        if distr_pos[1]:
            if self.verbose:
                logging.debug('Attached distr. before C2')
            temp = tf.concat([tf.zeros(s1.shape[2] - distr_to_concat_fc.shape[1]), self.y_distr], axis=0)
            distr_to_concat_c = tf.reshape(tf.tile(input=temp, multiples=[tf.shape(s1)[0]]),
                                           shape=[tf.shape(s1)[0], 1, int(temp.shape[0])])
            new_s1 = []
            for i in range(s1.shape[3]):
                new_s1.append(tf.concat([s1[:, :, :, i], distr_to_concat_c], axis=1))
            s1 = tf.stack(new_s1, axis=3)

        c2 = tf.nn.conv2d(s1, c2_weights, strides=[1, 1, 1, 1], padding='VALID') + c2_biases
        c2 = tf.nn.relu(c2)

        s2 = tf.nn.max_pool(c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # s2 = tf.nn.dropout(s2, self.keep_prob, seed=self.mnist_dataset.train.seed)
        s2 = flatten(s2)

        if distr_pos[2]:
            if self.verbose:
                logging.debug('Attached distr. before F1')
            s2 = tf.concat([s2, distr_to_concat_fc], axis=1)

        fc_layer_1_size = int(s2.shape[1])
        fc1_weights = tf.Variable(tf.truncated_normal(shape=(fc_layer_1_size, fc_layer_2_size), mean=mu, stddev=sigma))
        fc1_biases = tf.Variable(tf.zeros(fc_layer_2_size))
        fc1 = tf.matmul(s2, fc1_weights) + fc1_biases
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, self.keep_prob, seed=self.dataset.train.seed)

        if distr_pos[3]:
            if self.verbose:
                logging.debug('Attached distr. before F2')
            fc1 = tf.concat([fc1, distr_to_concat_fc], axis=1)

        fc_layer_2_size = int(fc1.shape[1])
        fc2_weights = tf.Variable(tf.truncated_normal(shape=(fc_layer_2_size, fc_layer_3_size), mean=mu, stddev=sigma))
        fc2_biases = tf.Variable(tf.zeros(fc_layer_3_size))
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, self.keep_prob, seed=self.dataset.train.seed)

        if distr_pos[4]:
            if self.verbose:
                logging.debug('Attached distr. before F3')
            fc2 = tf.concat([fc2, distr_to_concat_fc], axis=1)
            fc_layer_3_size += distr_size
        output_weights = tf.Variable(
            tf.truncated_normal(shape=(fc_layer_3_size, self.label_size), mean=mu, stddev=sigma))
        output_biases = tf.Variable(tf.zeros(self.label_size))
        logits = tf.matmul(fc2, output_weights) + output_biases

        if self.verbose:
            logging.info('Network layers size:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(
                x.get_shape().as_list(),
                c1.get_shape().as_list(),
                s1.get_shape().as_list(),
                c2.get_shape().as_list(),
                s2.get_shape().as_list(),
                fc1.get_shape().as_list(),
                fc2.get_shape().as_list(),
                logits.get_shape().as_list()))

        return logits

    def _mlp(self, x, no_color_channels, variable_mean, variable_stddev, distr_pos):
        """
        Implementation of simple MLP architecture, with the possibility of attaching batch distribution
        :param x: input images
        :param no_color_channels:
        :param variable_mean:
        :param variable_stddev:
        :return: the result of the last fully connected layer
        """
        # Hyperparameters
        distr_size = self.dataset.num_classes
        layer_1_size = self.dataset.image_size * self.dataset.image_size * no_color_channels
        layer_2_size = 128
        layer_3_size = 64
        output_layer_size = self.label_size
        mu = variable_mean
        sigma = variable_stddev

        # tile y_distr in order to attach it as input for network's layers
        distr_to_concat_fc = tf.reshape(tf.tile(input=self.y_distr, multiples=[tf.shape(x)[0]]),
                                        shape=[tf.shape(x)[0], distr_size])
        self.batch_distr = distr_to_concat_fc

        # implement network architecture
        flatten_input = flatten(x)

        if distr_pos[2]:
            if self.verbose:
                logging.debug('Attached distr. before F1')
            flatten_input = tf.concat([flatten_input, distr_to_concat_fc], axis=1)

        layer_1_size = int(flatten_input.shape[1])
        fc1_weights = tf.Variable(tf.truncated_normal(shape=(layer_1_size, layer_2_size), mean=mu, stddev=sigma))
        fc1_biases = tf.Variable(tf.zeros(layer_2_size))
        fc1 = tf.matmul(flatten_input, fc1_weights) + fc1_biases
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, self.keep_prob, seed=self.dataset.train.seed)

        if distr_pos[3]:
            if self.verbose:
                logging.debug('Attached distr. before F2')
            fc1 = tf.concat([fc1, distr_to_concat_fc], axis=1)

        layer_2_size = int(fc1.shape[1])
        fc2_weights = tf.Variable(tf.truncated_normal(shape=(layer_2_size, layer_3_size), mean=mu, stddev=sigma))
        fc2_biases = tf.Variable(tf.zeros(layer_3_size))
        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, self.keep_prob, seed=self.dataset.train.seed)

        if distr_pos[4]:
            if self.verbose:
                logging.debug('Attached distr. before F3')
            fc2 = tf.concat([fc2, distr_to_concat_fc], axis=1)
            layer_3_size += distr_size
        output_weights = tf.Variable(
            tf.truncated_normal(shape=(layer_3_size, output_layer_size), mean=mu, stddev=sigma))
        output_biases = tf.Variable(tf.zeros(output_layer_size))
        logits = tf.matmul(fc2, output_weights) + output_biases

        if self.verbose:
            logging.info('Network layers size:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(
                x.get_shape().as_list(),
                flatten_input.get_shape().as_list(),
                fc1.get_shape().as_list(),
                fc2.get_shape().as_list(),
                logits.get_shape().as_list()))

        return logits

    def eval_data(self, dataset, use_only_one_batch=True):
        if not use_only_one_batch:
            validation_batch_size = self.batch_size  # use train batch size
            steps_per_epoch = dataset.num_examples // validation_batch_size
            num_examples = steps_per_epoch * validation_batch_size
        else:
            validation_batch_size = dataset.num_examples
            steps_per_epoch = 1
            num_examples = dataset.num_examples
        total_acc, total_loss = 0, 0
        sess = self.session
        # tf.get_default_session()
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(validation_batch_size)
            batch_y_distr = np.bincount(np.argmax(batch_y, axis=1),
                                        minlength=self.dataset.num_classes) / batch_y.shape[0]
            loss, acc = sess.run([self.loss_op, self.accuracy_op], feed_dict={self.x: batch_x, self.y: batch_y,
                                                                              self.y_distr: batch_y_distr,
                                                                              self.keep_prob: 1.0})
            total_acc += (acc * batch_x.shape[0])
            total_loss += (loss * batch_x.shape[0])
        return total_loss / num_examples, total_acc / num_examples

    def test_data(self, dataset, use_only_one_batch=True, distr_to_attach=None, batch_size=100):
        if not use_only_one_batch:
            test_batch_size = batch_size
            steps_per_epoch = dataset.num_examples // test_batch_size
            num_examples = steps_per_epoch * test_batch_size
        else:
            test_batch_size = dataset.num_examples
            steps_per_epoch = 1
            num_examples = dataset.num_examples
        total_acc, total_loss = 0, 0
        total_predict, total_actual = [], []
        wrong_predict_images = []
        total_softmax_output_probs = None
        # tf.get_default_session()
        sess = self.session
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(test_batch_size)
            if distr_to_attach is None:
                batch_y_distr = np.bincount(np.argmax(batch_y, axis=1),
                                            minlength=self.dataset.num_classes) / batch_y.shape[0]
            else:
                batch_y_distr = distr_to_attach
            loss, acc, predict, actual, logits = sess.run(
                [self.loss_op, self.accuracy_op, tf.argmax(self.network, 1), tf.argmax(self.y, 1), self.network],
                feed_dict={self.x: batch_x, self.y: batch_y, self.y_distr: batch_y_distr, self.keep_prob: 1.0})

            total_acc += (acc * batch_x.shape[0])
            total_loss += (loss * batch_x.shape[0])
            total_predict = np.append(total_predict, predict)
            total_actual = np.append(total_actual, actual)
            softmax_output_probs = tf.Session().run(tf.nn.softmax(logits=logits))
            if total_softmax_output_probs is None:
                total_softmax_output_probs = softmax_output_probs
            else:
                total_softmax_output_probs = np.append(total_softmax_output_probs, softmax_output_probs, axis=0)
            for index in range(len(predict)):
                if predict[index] != actual[index]:
                    wrong_predict_images.append(batch_x[index])
        return total_loss / num_examples, total_acc / num_examples, total_predict.astype(np.int32), \
               total_actual.astype(np.int32), wrong_predict_images, total_softmax_output_probs

    def test_on_only_an_image(self, image, label, distr_to_attach=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                              get_output_probs=False):
        batch_x = image[None, :, :, :]
        batch_y = label[None, :]
        loss, acc, predict, actual, logits = self.session.run(
            [self.loss_op, self.accuracy_op, tf.argmax(self.network, 1), tf.argmax(self.y, 1), self.network],
            feed_dict={self.x: batch_x, self.y: batch_y, self.y_distr: distr_to_attach, self.keep_prob: 1.0})
        if get_output_probs:
            softmax_output_probs = tf.Session().run(tf.nn.softmax(logits=logits))
            return loss, acc, predict.astype(np.int32), actual.astype(np.int32), softmax_output_probs
        else:
            return loss, acc, predict.astype(np.int32), actual.astype(np.int32)

    def train(self, distrs_list=None, use_shuffling_inside_class=False, attach_imposed_distr=False):
        # reset epoch_completed and indices_in_epoch fields from mnist dataset
        # (in case if the same object is used for multiple trainings)
        self.dataset.train.reset_epochs_completed()
        self.dataset.train.reset_indices_in_epoch()
        saver = tf.train.Saver(save_relative_paths=True)
        if self.session is not None:
            self.session.close()
        with tf.Session() as self.session:
            self.session.run(tf.initialize_all_variables())
            steps_per_epoch = self.dataset.train.num_examples // self.batch_size
            num_examples = steps_per_epoch * self.batch_size
            logging.info('Training will use max. num_examples = {} from training set size = {}'
                         .format(num_examples, self.dataset.train.num_examples))

            # Train model
            for i in range(self.epochs):
                self.dataset.train.shuffle()
                total_tran_loss = 0.0
                total_tran_acc = 0.0
                # count how much examples are used effectively
                # because when imposing a distribution not all the data is used in an epoch
                concrete_num_examples_used_in_last_epoch = 0

                # start building batches with one distribution chosen randomly from distr_list
                if distrs_list is not None:
                    k = self.dataset.train.rg.randint(low=0, high=len(distrs_list))
                for step in range(steps_per_epoch):
                    if distrs_list is None:
                        batch_x, batch_y = self.dataset.train.next_batch(self.batch_size)
                    else:
                        distr_to_impose = distrs_list[k]
                        k = (k + 1) % len(distrs_list)
                        batch_x, batch_y = self.dataset.train.next_batch(self.batch_size, distr_to_impose,
                                                                         use_shuffling_inside_class)
                    batch_y_distr = np.bincount(np.argmax(batch_y, axis=1),
                                                minlength=self.dataset.num_classes) / batch_y.shape[0]

                    if attach_imposed_distr:
                        distr_to_attach = distr_to_impose
                    else:
                        distr_to_attach = batch_y_distr

                    _, train_loss, train_acc, batch_distr_out = self.session.run(
                        [self.train_op, self.loss_op, self.accuracy_op, self.batch_distr],
                        feed_dict={self.x: batch_x, self.y: batch_y, self.y_distr: distr_to_attach,
                                   self.keep_prob: self.drop_out_keep_prob})

                    total_tran_loss += (train_loss * batch_x.shape[0])
                    total_tran_acc += (train_acc * batch_x.shape[0])
                    concrete_num_examples_used_in_last_epoch += batch_x.shape[0]

                    # generating a batch w.r.t a distr. can cause finishing an epoch earlier
                    # if self.mnist_dataset.train.epochs_completed > i:
                    #     break

                total_tran_loss = total_tran_loss / concrete_num_examples_used_in_last_epoch
                total_tran_acc = total_tran_acc / concrete_num_examples_used_in_last_epoch
                val_loss, val_acc = self.eval_data(self.dataset.validation)
                logging.info(
                    "Epoch {:2d}/{:2d} --- Training: loss = {:.3f}, acc = {:.3f}; Validation: loss = {:.3f},"
                    " acc = {:.3f}; num_examples_used = {}"
                        .format(i + 1, self.epochs, total_tran_loss, total_tran_acc, val_loss, val_acc,
                                concrete_num_examples_used_in_last_epoch))
                self.plotter.add_loss_accuracy_to_plot(i, total_tran_loss, total_tran_acc, val_loss, val_acc,
                                                       redraw=True)

            saver.save(self.session, self.file_name_model)
            logging.info("Model saved into {}".format(self.file_name_model))

            # Evaluate on the test data
            test_loss, test_acc, total_predict, total_actual, wrong_predict_images, _ = self.test_data(
                self.dataset.test, use_only_one_batch=True)
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
                logging.error("Failed when plotting wrong predicted images:\n" + str(ex))
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
            logging.error('Directory {} not found.'.format(ckpt_dir))
        else:
            # train_distr was introduced later so we will try to restore as much we can from the checkpoint file
            reader = tf.train.NewCheckpointReader(os.path.join(dir, ckpt_filename))
            vars_to_restore = []
            for var in tf.global_variables():
                var_name = var.name.split(':')[0]
                if reader.has_tensor(var_name):
                    vars_to_restore.append(var)

            saver = tf.train.Saver(vars_to_restore)

            if self.session is not None:
                self.session.close()
            self.session = tf.Session()

            if ckpt_filename is not None:
                saver.restore(sess=self.session, save_path=os.path.join(dir, ckpt_filename))
            else:
                saver.restore(sess=self.session, save_path=tf.train.latest_checkpoint(ckpt_dir))
