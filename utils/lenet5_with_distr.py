import logging.config
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

from utils import utils
from utils.training_plotter import TrainingPlotter

import json

logging.config.fileConfig('logging.conf')


class Lenet5WithDistr(object):
    """
        Implements a Lenet5 architecture and methods for training, test and evaluation
        As input to different layers we also add the label distribution
    """

    NO_LAYERS = 5

    # default hyperparameters
    DEFAULT_SAVE_ROOT_DIR = './results/'
    DEFAULT_MODEL_NAME = ''
    DEFAULT_SHOW_PLOT_WINDOW = False
    DEFAULT_NO_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 500
    DEFAULT_VARIABLE_MEAN = 0.0
    DEFAULT_VARIABLE_STDDEV = 1.0
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_DROPOUT_KEEP_PROB = 1.0
    DEFAULT_VERBOSE = True
    DEFAULT_DISTR_POS = tuple([False] * NO_LAYERS)
    DEFAULT_DISTR_LIST = None
    DEFAULT_SHUFFLE_INSIDE_CLASS = False
    DEFAULT_ATTACH_IMPOSED_DISTR = False
    DEFAULT_SEED = 112358
    NO_DISTR_USED = -1

    def __init__(self, dataset, save_root_dir=DEFAULT_SAVE_ROOT_DIR, model_name=DEFAULT_MODEL_NAME,
                 show_plot_window=DEFAULT_SHOW_PLOT_WINDOW, verbose=DEFAULT_VERBOSE, epochs=DEFAULT_NO_EPOCHS,
                 batch_size=DEFAULT_BATCH_SIZE, variable_mean=DEFAULT_VARIABLE_MEAN,
                 variable_stddev=DEFAULT_VARIABLE_STDDEV, learning_rate=DEFAULT_LEARNING_RATE,
                 drop_out_keep_prob=DEFAULT_DROPOUT_KEEP_PROB, distr_pos=DEFAULT_DISTR_POS,
                 attach_imposed_distr=DEFAULT_ATTACH_IMPOSED_DISTR, distrs_list=DEFAULT_DISTR_LIST,
                 shuffle_inside_class=DEFAULT_SHUFFLE_INSIDE_CLASS, seed=DEFAULT_SEED):

        """
        :param dataset: the dataset containing train/validation/test subsets
        :param save_root_dir: the root directory where the training results will be stored
        :param model_name: a label assigned to a run instance
        :param show_plot_window: a flag for displaying or nor the training curse in real time
        :param verbose: a flag for displaying or not logs during training
        :param epochs: the number of epochs
        :param batch_size: the batch size
        :param variable_mean: the mean of the normal distribution used for weights initializing
        :param variable_stddev: the standard deviation of the normal distribution used for weights initializing
        :param learning_rate: the learning rate
        :param drop_out_keep_prob: the probability of keeping the units
        :param distr_pos: a boolean array (with the size given by number of layers) which specifies in which layers the
                          distribution will be concatenated as input;

        :param attach_imposed_distr: a boolean which is True if the theoretical distribution will be attached as input
                                     (i.e. the one imposed over the batch); otherwise the empirical one will be attached
                                     (i.e. the label distribution of the batch will be calculated)

        :param distrs_list: a list containing the distributions used for generating the batches
                           if None, random batches will be used

        :param shuffle_inside_class: parameter for impose_distribution method of Dataset class

        :param seed: the seed used for all random processes (e.g. generating the weights random initializations,
                     for dropout (if used) etc)
        """
        self.dataset = dataset

        # settings
        self.save_root_dir = save_root_dir
        self.model_name = model_name
        self.show_plot_window = show_plot_window
        self.verbose = verbose
        self.seed = seed

        # set hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.variable_mean = variable_mean
        self.variable_stddev = variable_stddev
        self.learning_rate = learning_rate
        self.drop_out_keep_prob = drop_out_keep_prob
        self.distr_pos = distr_pos
        self.attach_imposed_distr = attach_imposed_distr
        self.distrs_list = distrs_list
        self.shuffle_inside_class = shuffle_inside_class

        # use a timestamp to identify the current run instance
        self.ts = utils.now_as_ts()

        # set other useful variables
        self.num_color_channels = self.dataset.train.images.shape[3]
        self.num_classes = self.dataset.num_classes
        self.labels_name = [str(i) for i in range(self.num_classes)]

        self._prepare_files()

        # build the computational graph
        self.graph = self.TfGraph(self)

        # set the current session
        self.session = None

        # display the summary of the dataset
        if self.verbose:
            logging.info(dataset.summary)

    def _prepare_files(self):
        # check if the save root directory exists, otherwise create it
        if not os.path.isdir(self.save_root_dir):
            os.mkdir(self.save_root_dir)

        # make a dir inside the save root directory for the current run instance
        self.save_dir = os.path.join(self.save_root_dir, str(self.ts))
        os.mkdir(self.save_dir)

        # prepare all necessary files for saving the learning process results
        base_file_name = '{}/{}_{}_{}'.format(self.save_dir, self.__class__.__name__, self.model_name,
                                              utils.now_as_str())
        self.file_name_learning_curve = '{}.learning_curve.png'.format(base_file_name)
        self.file_name_model = '{}.model.ckpt'.format(base_file_name)
        self.file_name_confusion_matrix = '{}.confusion_matrix.png'.format(base_file_name)
        self.file_name_wrong_predicts = '{}.wrong_predicts.png'.format(base_file_name)
        self.plotter = TrainingPlotter(title="{}_{}".format(self.__class__.__name__, self.model_name),
                                       file_name=self.file_name_learning_curve, show_plot_window=self.show_plot_window)

    @staticmethod
    def get_run_with_settings(dataset, settings):
        """
        :param dataset: the dataset containing train/validation/test data
        :param settings: a dictionary containing the settings (e.g. save_dir, model_name)
                         and the hyperparameters for the Lenet5WithDistr model
        :return: an instance of the model with the given parameters
        # """

        # read settings
        seed = Lenet5WithDistr._get_value(settings, 'seed', Lenet5WithDistr.DEFAULT_SEED)
        save_root_dir = Lenet5WithDistr._get_value(settings, 'save_root_dir', Lenet5WithDistr.DEFAULT_SAVE_ROOT_DIR)
        model_name = Lenet5WithDistr._get_value(settings, 'model_name', Lenet5WithDistr.DEFAULT_MODEL_NAME)
        show_plot_window = Lenet5WithDistr._get_value(settings, 'show_plot_window',
                                                      Lenet5WithDistr.DEFAULT_SHOW_PLOT_WINDOW)
        verbose = Lenet5WithDistr._get_value(settings, 'verbose', Lenet5WithDistr.DEFAULT_VERBOSE)

        # read hyperparameters
        hp = Lenet5WithDistr._get_value(settings, 'hyperparameters', None)
        if hp is None:
            logging.warning('All hyperparameters will be set to default.')
            return Lenet5WithDistr(dataset, save_root_dir=save_root_dir, model_name=model_name,
                                   show_plot_window=show_plot_window, verbose=verbose)

        epochs = Lenet5WithDistr._get_value(hp, 'epochs', Lenet5WithDistr.DEFAULT_NO_EPOCHS)
        batch_size = Lenet5WithDistr._get_value(hp, 'batch_size', Lenet5WithDistr.DEFAULT_BATCH_SIZE)
        variable_mean = Lenet5WithDistr._get_value(hp, 'variable_mean', Lenet5WithDistr.DEFAULT_VARIABLE_MEAN)
        variable_stddev = Lenet5WithDistr._get_value(hp, 'variable_stddev', Lenet5WithDistr.DEFAULT_VARIABLE_STDDEV)
        learning_rate = Lenet5WithDistr._get_value(hp, 'learning_rate', Lenet5WithDistr.DEFAULT_LEARNING_RATE)
        drop_out_keep_prob = Lenet5WithDistr._get_value(hp, 'drop_out_keep_prob',
                                                        Lenet5WithDistr.DEFAULT_DROPOUT_KEEP_PROB)
        distr_pos = Lenet5WithDistr._get_value(hp, 'distr_pos', Lenet5WithDistr.DEFAULT_DISTR_POS)
        attach_imposed_distr = Lenet5WithDistr._get_value(hp, 'attach_imposed_distr',
                                                          Lenet5WithDistr.DEFAULT_ATTACH_IMPOSED_DISTR)
        distrs_list = Lenet5WithDistr._get_value(hp, 'distrs_list', Lenet5WithDistr.DEFAULT_DISTR_LIST)
        shuffle_inside_class = Lenet5WithDistr._get_value(hp, 'shuffle_inside_class',
                                                          Lenet5WithDistr.DEFAULT_SHUFFLE_INSIDE_CLASS)

        return Lenet5WithDistr(dataset, save_root_dir=save_root_dir, model_name=model_name,
                               show_plot_window=show_plot_window, verbose=verbose, epochs=epochs, batch_size=batch_size,
                               variable_mean=variable_mean, variable_stddev=variable_stddev,
                               learning_rate=learning_rate, drop_out_keep_prob=drop_out_keep_prob, distr_pos=distr_pos,
                               attach_imposed_distr=attach_imposed_distr, distrs_list=distrs_list,
                               shuffle_inside_class=shuffle_inside_class, seed=seed)

    @staticmethod
    def _get_value(settings, key, default):
        try:
            return settings[key]
        except KeyError:
            logging.warning("Key '{}' not found in settings; the default value ({}) will be used.".format(key, default))
            return default

    class TfGraph:
        """
        Implements all the operations in the current graph, corresponding to Lenet5 architecture and appends the
        distribution in the layers specified (if any).
        """

        def __init__(self, l5):
            # clear the default graph
            tf.reset_default_graph()

            # set input variables
            self.input_image = tf.placeholder(tf.float32, (None, l5.dataset.image_size, l5.dataset.image_size,
                                                           l5.num_color_channels))
            self.y = tf.placeholder(tf.float32, (None, l5.dataset.num_classes))
            self.train_distr = tf.Variable(initial_value=l5.dataset.train.label_distr, name='train_distr')
            self.test_distr = tf.Variable(initial_value=l5.dataset.test.label_distr, name='test_distr')
            self.distr_pos = tf.Variable(initial_value=l5.distr_pos, name='distr_pos')
            self.train_num_examples = tf.Variable(initial_value=l5.dataset.train.num_examples,
                                                  name='train_num_examples')
            self.y_distr = tf.placeholder(tf.float32, (l5.num_classes,), name='y_distr')
            self.keep_prob = tf.placeholder(tf.float32)

            # build intput to output operations
            _patch_size = 5
            _conv_layer_1_depth = 6
            _conv_layer_2_depth = 16
            _fc_layer_1_size = 400
            _fc_layer_2_size = 120
            _fc_layer_3_size = 84
            _mu = l5.variable_mean
            _sigma = l5.variable_stddev

            # tile y_distr in order to attach it as input for network's layers
            self.batch_distr = tf.reshape(tf.tile(input=self.y_distr, multiples=[tf.shape(self.input_image)[0]]),
                                          shape=[tf.shape(self.input_image)[0], l5.num_classes])
            self.c1_weights = tf.Variable(
                tf.truncated_normal(shape=(_patch_size, _patch_size, l5.num_color_channels, _conv_layer_1_depth),
                                    mean=_mu, stddev=_sigma, seed=l5.seed))
            self.c1_biases = tf.Variable(tf.zeros(_conv_layer_1_depth))
            if l5.distr_pos[0]:
                if l5.verbose:
                    logging.debug('Attached distr. before C1 (at input)')
                temp = tf.concat([tf.zeros(self.input_image.shape[2] - self.batch_distr.shape[1]), self.y_distr],
                                 axis=0)
                distr_to_concat_c = tf.reshape(tf.tile(input=temp, multiples=[tf.shape(self.input_image)[0]]),
                                               shape=[tf.shape(self.input_image)[0], 1, int(temp.shape[0]), 1])
                self.tf_all_inputs = tf.concat([self.input_image, distr_to_concat_c], axis=1)
            else:
                self.tf_all_inputs = self.input_image

            self.c1 = tf.nn.conv2d(self.tf_all_inputs, self.c1_weights, strides=[1, 1, 1, 1],
                                   padding='SAME') + self.c1_biases
            self.c1 = tf.nn.relu(self.c1)
            self.s1 = tf.nn.max_pool(self.c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.c2_weights = tf.Variable(
                tf.truncated_normal(shape=(_patch_size, _patch_size, _conv_layer_1_depth, _conv_layer_2_depth),
                                    mean=_mu, stddev=_sigma, seed=l5.seed))
            self.c2_biases = tf.Variable(tf.zeros(_conv_layer_2_depth))

            if l5.distr_pos[1]:
                if l5.verbose:
                    logging.debug('Attached distr. before C2')
                temp = tf.concat([tf.zeros(self.s1.shape[2] - self.batch_distr.shape[1]), self.y_distr], axis=0)
                distr_to_concat_c = tf.reshape(tf.tile(input=temp, multiples=[tf.shape(self.s1)[0]]),
                                               shape=[tf.shape(self.s1)[0], 1, int(temp.shape[0])])
                new_s1 = []
                for i in range(self.s1.shape[3]):
                    new_s1.append(tf.concat([self.s1[:, :, :, i], distr_to_concat_c], axis=1))
                self.s1 = tf.stack(new_s1, axis=3)
            self.c2 = tf.nn.conv2d(self.s1, self.c2_weights, strides=[1, 1, 1, 1], padding='VALID') + self.c2_biases
            self.c2 = tf.nn.relu(self.c2)
            self.s2 = tf.nn.max_pool(self.c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            self.s2 = flatten(self.s2)

            if l5.distr_pos[2]:
                if l5.verbose:
                    logging.debug('Attached distr. before F1')
                self.s2 = tf.concat([self.s2, self.batch_distr], axis=1)

            _fc_layer_1_size = int(self.s2.shape[1])
            self.fc1_weights = tf.Variable(
                tf.truncated_normal(shape=(_fc_layer_1_size, _fc_layer_2_size), mean=_mu, stddev=_sigma, seed=l5.seed))
            self.fc1_biases = tf.Variable(tf.zeros(_fc_layer_2_size))
            self.fc1 = tf.matmul(self.s2, self.fc1_weights) + self.fc1_biases
            self.fc1 = tf.nn.relu(self.fc1)
            self.fc1 = tf.nn.dropout(self.fc1, self.keep_prob, seed=l5.seed)

            if l5.distr_pos[3]:
                if l5.verbose:
                    logging.debug('Attached distr. before F2')
                self.fc1 = tf.concat([self.fc1, self.batch_distr], axis=1)

            _fc_layer_2_size = int(self.fc1.shape[1])
            self.fc2_weights = tf.Variable(
                tf.truncated_normal(shape=(_fc_layer_2_size, _fc_layer_3_size), mean=_mu, stddev=_sigma, seed=l5.seed))
            self.fc2_biases = tf.Variable(tf.zeros(_fc_layer_3_size))
            self.fc2 = tf.matmul(self.fc1, self.fc2_weights) + self.fc2_biases
            self.fc2 = tf.nn.relu(self.fc2)
            self.fc2 = tf.nn.dropout(self.fc2, self.keep_prob, seed=l5.seed)

            if l5.distr_pos[4]:
                if l5.verbose:
                    logging.debug('Attached distr. before F3')
                self.fc2 = tf.concat([self.fc2, self.batch_distr], axis=1)
                _fc_layer_3_size += l5.num_classes
            self.output_weights = tf.Variable(
                tf.truncated_normal(shape=(_fc_layer_3_size, l5.dataset.num_classes), mean=_mu, stddev=_sigma,
                                    seed=l5.seed))
            output_biases = tf.Variable(tf.zeros(l5.dataset.num_classes))
            self.tf_logits = tf.matmul(self.fc2, self.output_weights) + output_biases

            if l5.verbose:
                logging.info('Network layers size:\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(
                    self.input_image.get_shape().as_list(),
                    self.c1.get_shape().as_list(),
                    self.s1.get_shape().as_list(),
                    self.c2.get_shape().as_list(),
                    self.s2.get_shape().as_list(),
                    self.c1.get_shape().as_list(),
                    self.fc2.get_shape().as_list(),
                    self.tf_logits.get_shape().as_list()))

            # set optimizing settings
            self.prediction_softmax = tf.nn.softmax(self.tf_logits)
            self.loss_op = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.tf_logits))
            self.opt = tf.train.AdamOptimizer(learning_rate=l5.learning_rate)
            self.train_loss_opt = self.opt.minimize(self.loss_op)
            self.correct_prediction = tf.equal(tf.argmax(self.tf_logits, 1), tf.argmax(self.y, 1))
            self.accuracy_op = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

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
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(validation_batch_size)
            batch_y_distr = np.bincount(np.argmax(batch_y, axis=1),
                                        minlength=self.num_classes) / batch_y.shape[0]
            loss, acc = sess.run([self.graph.loss_op, self.graph.accuracy_op],
                                 feed_dict={self.graph.input_image: batch_x, self.graph.y: batch_y,
                                            self.graph.y_distr: batch_y_distr,
                                            self.graph.keep_prob: 1.0})
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
        sess = self.session
        for step in range(steps_per_epoch):
            batch_x, batch_y = dataset.next_batch(test_batch_size)
            if distr_to_attach is None:
                batch_y_distr = np.bincount(np.argmax(batch_y, axis=1),
                                            minlength=self.num_classes) / batch_y.shape[0]
            else:
                batch_y_distr = distr_to_attach
            loss, acc, predict, actual, logits = sess.run(
                [self.graph.loss_op, self.graph.accuracy_op, tf.argmax(self.graph.tf_logits, 1),
                 tf.argmax(self.graph.y, 1),
                 self.graph.tf_logits],
                feed_dict={self.graph.input_image: batch_x, self.graph.y: batch_y, self.graph.y_distr: batch_y_distr,
                           self.graph.keep_prob: 1.0})

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
        return total_loss / num_examples, total_acc / num_examples, total_predict.astype(np.int32), total_actual.astype(
            np.int32), wrong_predict_images, total_softmax_output_probs

    def test_on_only_an_image(self, image, label, distr_to_attach=tuple([0.1] * 10), get_output_probs=False):
        batch_x = image[None, :, :, :]
        batch_y = label[None, :]
        loss, acc, predict, actual, logits = self.session.run(
            [self.graph.loss_op, self.graph.accuracy_op, tf.argmax(self.graph.tf_logits, 1),
             tf.argmax(self.graph.y, 1),
             self.graph.tf_logits],
            feed_dict={self.graph.input_image: batch_x, self.graph.y: batch_y, self.graph.y_distr: distr_to_attach,
                       self.graph.keep_prob: 1.0})
        if get_output_probs:
            softmax_output_probs = tf.Session().run(tf.nn.softmax(logits=logits))
            return loss, acc, predict.astype(np.int32), actual.astype(np.int32), softmax_output_probs
        else:
            return loss, acc, predict.astype(np.int32), actual.astype(np.int32)

    def train(self):
        # reset epoch_completed and indices_in_epoch fields from dataset
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
                # count how many examples are used effectively
                # because when imposing a distribution not all the data is used in an epoch
                concrete_num_examples_used_in_last_epoch = 0

                # start building batches with one distribution chosen randomly from distr_list
                k = None
                distr_to_impose = None
                if self.distrs_list is not None:
                    k = self.dataset.train.rg.randint(low=0, high=len(self.distrs_list))
                for step in range(steps_per_epoch):
                    if self.distrs_list is None:
                        batch_x, batch_y = self.dataset.train.next_batch(self.batch_size)
                    else:
                        distr_to_impose = self.distrs_list[k]
                        k = (k + 1) % len(self.distrs_list)
                        batch_x, batch_y = self.dataset.train.next_batch(self.batch_size, distr_to_impose,
                                                                         self.shuffle_inside_class)
                    batch_y_distr = np.bincount(np.argmax(batch_y, axis=1),
                                                minlength=self.num_classes) / batch_y.shape[0]

                    if self.attach_imposed_distr:
                        distr_to_attach = distr_to_impose
                    else:
                        distr_to_attach = batch_y_distr

                    _, train_loss, train_acc, batch_distr_out = self.session.run(
                        [self.graph.train_loss_opt, self.graph.loss_op, self.graph.accuracy_op, self.graph.batch_distr],
                        feed_dict={self.graph.input_image: batch_x, self.graph.y: batch_y,
                                   self.graph.y_distr: distr_to_attach,
                                   self.graph.keep_prob: self.drop_out_keep_prob})

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
                    " acc = {:.3f}; num_examples_used = {}".format(i + 1, self.epochs, total_tran_loss, total_tran_acc,
                                                                   val_loss, val_acc,
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
                wrong_predict_images_sorted = wrong_predict_images[wrong_actual.argsort()]
                wrong_predict_images_sorted = [image for image in wrong_predict_images_sorted]
                self.plotter.combine_images(wrong_predict_images_sorted, self.file_name_wrong_predicts)
            except Exception as ex:
                logging.error("Failed when plotting wrong predicted images:\n" + str(ex))

            self.plotter.safe_shut_down()

            # save all run instance details
            self._save_run_instance_details()

    def _save_run_instance_details(self):
        # gather all hyperparameters
        hp = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'variable_mean': self.variable_mean,
            'variable_stddev': self.variable_stddev,
            'learning_rate': self.learning_rate,
            'drop_out_keep_prob': self.drop_out_keep_prob,
            'distr_pos': list(self.distr_pos),
            'attach_imposed_distr': self.attach_imposed_distr,
            'distrs_list': [list(distr) for distr in self.distrs_list],
            'shuffle_inside_class': self.shuffle_inside_class
        }

        # gather all settings
        settings = {
            'seed': self.seed,
            'save_root_dir': self.save_root_dir,
            'model_name': self.model_name,
            'show_plot_window': self.show_plot_window,
            'verbose': self.verbose,
            'hyperparameters': hp
        }

        # save performance metrics
        train_loss, train_acc = self.eval_data(self.dataset.train)
        validation_loss, validation_acc = self.eval_data(self.dataset.validation)
        test_loss, test_acc = self.eval_data(self.dataset.test)

        perf = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'validation_loss': validation_loss,
            'validation_acc': validation_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        }

        to_save = {
            'timestamp': self.ts,
            'performance_metrics': perf,
            'dataset_summary': self.dataset.summary,
            'settings': settings,
        }

        fn = '{}/run_details.json'.format(self.save_dir)
        with open(fn, 'w') as fp:
            json.dump(to_save, fp, indent=4)
            logging.info("Run instance '{}' details successfully saved to file '{}'.".format(self.ts, fn))

    def predict_images(self, images):
        return self.session.run(self.graph.prediction_softmax,
                                feed_dict={self.graph.input_image: images, self.graph.keep_prob: 1.0})

    def restore_session(self, ckpt_dir, ckpt_filename=None):
        """
        Function for restore a model session from previous saved ones

        :param ckpt_dir: a directory for checkpoint to search in
        :param ckpt_filename: try to restore model with this checkpoint file name; if is None, restore a model using
                              latest_checkpoint method from ckpt_dir directory
        """
        _dir = os.path.dirname(ckpt_dir)

        # check if directory ckpt_dir exists
        if not os.path.exists(_dir):
            logging.error('Directory {} not found.'.format(ckpt_dir))
        else:
            # train_distr was introduced later so we will try to restore as much we can from the checkpoint file
            reader = tf.train.NewCheckpointReader(os.path.join(_dir, ckpt_filename))
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
                saver.restore(sess=self.session, save_path=os.path.join(_dir, ckpt_filename))
            else:
                saver.restore(sess=self.session, save_path=tf.train.latest_checkpoint(ckpt_dir))
