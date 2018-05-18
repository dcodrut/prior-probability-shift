from sklearn.model_selection import train_test_split
from utils import Utils
from enhance_data import *
import numpy as np


class Dataset(object):
    seed = 112358
    rg = np.random.RandomState(seed)

    def __init__(self, images, labels, num_classes):
        assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._num_classes = num_classes

        # the following variables are used to generate a batch w.r.t. a given label distribution
        self._index_in_epoch = 0
        self._indices_in_epoch = np.zeros(num_classes, np.int32)
        self._indices_per_class = np.arange(num_classes) == np.argmax(labels, axis=1)[:, np.newaxis]
        self._counts_per_class = np.sum(self._indices_per_class, axis=0)
        self._label_distr = self._counts_per_class / np.sum(self._counts_per_class)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def counts_per_class(self):
        return self._counts_per_class

    @property
    def label_distr(self):
        return self._label_distr

    def reset_indices_in_epoch(self):
        self._indices_in_epoch = np.zeros(self.num_classes, np.int32)

    @staticmethod
    def reset_rg():
        Dataset.rg = np.random.RandomState(Dataset.seed)

    def reset_epochs_completed(self):
        self._epochs_completed = 0

    def shuffle(self):
        perm = np.arange(self._num_examples)
        Dataset.rg.shuffle(perm)
        self._images = self._images[perm,]
        self._labels = self._labels[perm,]

        # Data was shuffled => must recompute indices per class
        self._indices_per_class = np.arange(self._num_classes) == np.argmax(self.labels, axis=1)[:, np.newaxis]

    def next_batch(self, batch_size, weights=None):
        """
        Returns the next `batch_size` examples from this data set.
        If weights parameter is given, then the generated batch will respect that distribution (of labels).
        Otherwise, an uniform distribution is assumed.

        :param batch_size: how many examples the batch will contain
        :param weights: importance of each label
        :return: batch of examples and their labels
        """
        if weights is None:
            start = self._index_in_epoch
            self._index_in_epoch += batch_size
            if self._index_in_epoch > self._num_examples:
                # Finished epoch
                self._epochs_completed += 1

                # Shuffle the data
                self.shuffle()

                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
                assert batch_size <= self._num_examples
            end = self._index_in_epoch
            batch_x = self._images[start:end]
            batch_y = self._labels[start:end]
        else:
            # make sure that weights sum to 1
            weights = np.array(weights)
            weights = weights / sum(weights)
            start_indices = np.empty_like(self._indices_in_epoch)
            np.copyto(start_indices, self._indices_in_epoch)
            num_examples_from_each_class = np.floor(weights * batch_size).astype(np.int32)

            # if we don't have batch_size examples, add the remaining ones to the most weighted class
            diff = batch_size - np.sum(num_examples_from_each_class)
            if diff > 0:
                num_examples_from_each_class[np.argmax(weights)] += diff
            # print('Imposing distr: num_examples_from_each_class = ', num_examples_from_each_class)
            self._indices_in_epoch += num_examples_from_each_class
            if np.sum(self._indices_in_epoch > self._counts_per_class) > 0:
                # Finished epoch
                self._epochs_completed += 1

                # Shuffle the data
                self.shuffle()

                # Start next epoch
                start_indices = np.zeros(self.num_classes, np.int32)
                self._indices_in_epoch = num_examples_from_each_class

                # Check if there are enough images of each label to build a batch with given distribution
                assert np.sum(num_examples_from_each_class > self._counts_per_class) == 0, \
                    'Not enough images of each label to build a batch of size ' + str(batch_size) + \
                    ' with given distribution: ' + \
                    '\n\tCounts per class in dataset = ' + str(self._counts_per_class) + \
                    '\n\tCounts per class needed to build the batch = ' + str(num_examples_from_each_class)

            end_indices = self._indices_in_epoch
            # print(end_indices)
            # print(sum(end_indices))
            batch_x = np.empty((batch_size,) + self._images[0].shape)
            batch_y = np.empty((batch_size, self._num_classes))
            start_index_in_batch = 0
            # print('start_indices = ', start_indices)
            # print('end_indices = ', end_indices)
            for i in range(self.num_classes):
                if num_examples_from_each_class[i] > 0:
                    images_of_class_i = self._images[self._indices_per_class[:, i],]
                    labels_of_class_i = self._labels[self._indices_per_class[:, i],]
                    end_index_in_batch = start_index_in_batch + num_examples_from_each_class[i]
                    batch_x[start_index_in_batch:end_index_in_batch, ] = \
                        images_of_class_i[start_indices[i]:end_indices[i], ]
                    batch_y[start_index_in_batch:end_index_in_batch] = labels_of_class_i[
                                                                       start_indices[i]:end_indices[i]]
                    start_index_in_batch = end_index_in_batch

            # shuffle images inside batch because they're ordered by label
            perm = np.arange(batch_size)
            Dataset.rg.shuffle(perm)
            batch_x = batch_x[perm,]
            batch_y = batch_y[perm,]

        return batch_x, batch_y


class MNISTDataset(object):
    image_size = 28
    num_channels = 1  # grayscale
    num_classes = 10

    def __init__(self, train_images_path, train_labels_path, test_images_path, test_labels_path):
        train_validation_size = 60000
        train_size = 54000  # number of images and labels from train dataset
        validation_size = 6000  # number of images and labels from validation dataset
        test_size = 10000  # number of images and labels from test dataset

        train_validation_images = MNISTDataset.load_images(train_images_path, train_validation_size)
        train_validation_labels = MNISTDataset.load_labels(train_labels_path, train_validation_size)
        test_images = MNISTDataset.load_images(test_images_path, test_size)
        test_labels = MNISTDataset.load_labels(test_labels_path, test_size)
        train_images, validation_images, train_labels, validation_labels = \
            train_test_split(train_validation_images, train_validation_labels,
                             test_size=validation_size, random_state=42)

        # convert labels into one-hot vectors
        train_labels = Utils.dense_to_one_hot(train_labels, MNISTDataset.num_classes)
        validation_labels = Utils.dense_to_one_hot(validation_labels, MNISTDataset.num_classes)
        test_labels = Utils.dense_to_one_hot(test_labels, MNISTDataset.num_classes)

        self._train = Dataset(train_images, train_labels, MNISTDataset.num_classes)
        self._validation = Dataset(validation_images, validation_labels, MNISTDataset.num_classes)
        self._test = Dataset(test_images, test_labels, MNISTDataset.num_classes)

        # save a screenshot of current dataset (maybe used in impose_distribution method)
        self.backup()

        # reset Dataset's random generator
        Dataset.reset_rg()

    @staticmethod
    def load_images(filepath, num_images):
        with open(filepath, 'rb') as f:
            f.read(16)  # skip magic number, number of images, number of rows, number of columns
            buf = f.read(MNISTDataset.image_size * MNISTDataset.image_size * num_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            images = data.reshape(num_images, MNISTDataset.image_size, MNISTDataset.image_size,
                                  MNISTDataset.num_channels) / 255.0
            return images

    @staticmethod
    def load_labels(filepath, num_labels):
        with open(filepath, 'rb') as f:
            f.read(8)  # skip magic number and number of labels
            buf = f.read(num_labels)
            labels = np.frombuffer(buf, dtype=np.uint8)
            return labels

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test

    def enhance_with_random_rotate(self, ratio=1):
        enh_train_images, enh_train_labels = enhance_with_random_rotate(self.train.images,
                                                                        np.argmax(self.train.labels, axis=1),
                                                                        ratio)
        # reformat labels into one-hot vectors
        enh_train_labels = Utils.dense_to_one_hot(enh_train_labels, MNISTDataset.num_classes)
        self._train = Dataset(enh_train_images, enh_train_labels, MNISTDataset.num_classes)

    def enhance_with_random_zoomin(self, ratio=1):
        enh_train_images, enh_train_labels = enhance_with_random_zoomin(self.train.images,
                                                                        np.argmax(self.train.labels, axis=1),
                                                                        ratio)
        # reformat labels into one-hot vectors
        enh_train_labels = Utils.dense_to_one_hot(enh_train_labels, MNISTDataset.num_classes)
        self._train = Dataset(enh_train_images, enh_train_labels, MNISTDataset.num_classes)

    def enhance_with_random_zoomin_and_rotate(self, ratio=1):
        enh_train_images, enh_train_labels = enhance_with_random_zoomin_and_rotate(self.train.images,
                                                                                   np.argmax(self.train.labels, axis=1),
                                                                                   ratio)
        # reformat labels into one-hot vectors
        enh_train_labels = Utils.dense_to_one_hot(enh_train_labels, MNISTDataset.num_classes)
        self._train = Dataset(enh_train_images, enh_train_labels, MNISTDataset.num_classes)

    @property
    def summary(self):
        return """
        training data set: images = {}, labels = {}, distr = {}
        validation data set: images = {}, labels = {}, distr = {}
        testing data set: images = {}, labels = {}, distr = {}
        """.format(self.train.images.shape, self.train.labels.shape,
                   np.round(self.train.label_distr, decimals=3),
                   self.validation.images.shape, self._validation.labels.shape,
                   np.round(self.validation.label_distr, decimals=3),
                   self.test.images.shape, self.test.labels.shape,
                   np.round(self.test.label_distr, decimals=3)
                   )

    def impose_distribution(self, weights, global_max_weight=None, max_training_size=None):
        """
        Overwrites current MNIST dataset with a subset w.r.t. a given distribution

        :param weights: label distribution
        :param global_max_weight: - if multiple distributions will be considered in training, than we might need the
                                global maximum weight value in order to build subsets of the same size for
                                all distributions considered, so we need to take it into account when building the
                                subset
                                 - if it's None, than global_max_weight will be local maximum (i.e. the maximum value
                                 of the current weights considered)
        """

        if global_max_weight is None:
            max_weight = np.max(weights)
        else:
            max_weight = global_max_weight

        # restore original dataset and reset start indices in order to start building the subset from the beginning
        self.restore_from_backup()
        self.train.reset_indices_in_epoch()
        self.validation.reset_indices_in_epoch()
        self.test.reset_indices_in_epoch()

        train_num_examples = np.floor(np.min(self.train.counts_per_class) / max_weight).astype(np.int32)
        if max_training_size is not None and train_num_examples > max_training_size:
            train_num_examples = max_training_size
        # round to hundreds
        train_num_examples -= train_num_examples % 100
        train_subset_x, train_subset_y = self.train.next_batch(batch_size=train_num_examples, weights=weights)

        validation_num_examples = np.floor(np.min(self.validation.counts_per_class) / max_weight).astype(np.int32)
        # round to hundreds
        validation_num_examples -= validation_num_examples % 100
        validation_subset_x, validation_subset_y = self.validation.next_batch(batch_size=validation_num_examples,
                                                                              weights=weights)

        test_num_examples = np.floor(np.min(self.test.counts_per_class) / max_weight).astype(np.int32)
        # round to hundreds
        test_num_examples -= test_num_examples % 100
        test_subset_x, test_subset_y = self.test.next_batch(batch_size=test_num_examples, weights=weights)

        self._train = Dataset(train_subset_x, train_subset_y, MNISTDataset.num_classes)
        self._validation = Dataset(validation_subset_x, validation_subset_y, MNISTDataset.num_classes)
        self._test = Dataset(test_subset_x, test_subset_y, MNISTDataset.num_classes)

    def impose_distr_on_train_dataset(self, subset_size, weights):
        train_subset_x, train_subset_y = self.train.next_batch(batch_size=subset_size, weights=weights)
        self._train = Dataset(train_subset_x, train_subset_y, MNISTDataset.num_classes)
        self.backup()  # in order to overwrite the entire dataset

    def backup(self):
        self._train_backup = self._train
        self._validation_backup = self._validation
        self._test_backup = self._test

    def restore_from_backup(self):
        self._train = self._train_backup
        self._validation = self._validation_backup
        self._test = self._test_backup
