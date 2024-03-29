import pickle

from sklearn.model_selection import train_test_split

import utils
from enhance_data import *


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

        # the following variables are used to generate a batch (simple or w.r.t. a given label distribution)
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

    @property
    def indices_per_class(self):
        return self._indices_per_class

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
        self._images = self._images[perm]
        self._labels = self._labels[perm]

        # Data was shuffled => must recompute indices per class
        self._indices_per_class = np.arange(self._num_classes) == np.argmax(self.labels, axis=1)[:, np.newaxis]

    def shuffle_only_a_class(self, c):
        perm = np.arange(self._counts_per_class[c])
        Dataset.rg.shuffle(perm)
        self._images[self._indices_per_class[:, c]] = self._images[self._indices_per_class[:, c]][perm]
        self._labels[self._indices_per_class[:, c]] = self._labels[self._indices_per_class[:, c]][perm]

        # no need to recompute indices per class (data was shuffled only inside a class)

    def next_batch(self, batch_size, weights=None, use_shuffling_inside_class=False):
        """
        Returns the next `batch_size` examples from this data set.
        If weights parameter is given, then the generated batch will respect that distribution (of labels).
        Otherwise, an uniform distribution is assumed.

        :param batch_size: how many examples the batch will contain
        :param weights: importance of each label
        :param use_shuffling_inside_class: if True, when all images from a class were running out, only those
               are shuffled, not the entire dataset
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

            # if we don't have batch_size examples, share the remaining ones, starting with the most weighted class
            diff = batch_size - np.sum(num_examples_from_each_class)
            # print('diff = ', diff)
            if diff > 0:
                indices_sorted_weights = np.argsort(-weights)  # sort descending
                k = 0
                while diff > 0:
                    num_examples_from_each_class[indices_sorted_weights[k]] += 1
                    diff -= 1
                    k = (k + 1) % len(weights)

            self._indices_in_epoch += num_examples_from_each_class
            if np.sum(self._indices_in_epoch > self._counts_per_class) > 0:
                # Finished epoch
                self._epochs_completed += 1

                # Shuffle the data
                if not use_shuffling_inside_class:
                    self.shuffle()

                    # Start next epoch
                    start_indices = np.zeros(self.num_classes, np.int32)
                    self._indices_in_epoch = num_examples_from_each_class

                else:
                    # shuffle only the data corresponding to the classes that are finished
                    for c in range(self.num_classes):
                        if self._indices_in_epoch[c] > self._counts_per_class[c]:
                            self.shuffle_only_a_class(c)
                            start_indices[c] = 0
                            self._indices_in_epoch[c] = num_examples_from_each_class[c]

                # Check if there are enough images of each label to build a batch with given distribution
                assert np.sum(num_examples_from_each_class > self._counts_per_class) == 0, \
                    'Not enough images of each label to build a batch of size ' + str(batch_size) + \
                    ' with given distribution: ' + \
                    '\n\tCounts per class in dataset = ' + str(self._counts_per_class) + \
                    '\n\tCounts per class needed to build the batch = ' + str(num_examples_from_each_class)

            end_indices = self._indices_in_epoch
            batch_x = np.empty((batch_size,) + self._images[0].shape, dtype=np.float32)
            batch_y = np.empty((batch_size, self._num_classes))
            start_index_in_batch = 0

            for i in range(self.num_classes):
                if num_examples_from_each_class[i] > 0:
                    images_of_class_i = self._images[self._indices_per_class[:, i]]
                    labels_of_class_i = self._labels[self._indices_per_class[:, i]]
                    end_index_in_batch = start_index_in_batch + num_examples_from_each_class[i]
                    batch_x[start_index_in_batch:end_index_in_batch, ] = \
                        images_of_class_i[start_indices[i]:end_indices[i], ]
                    batch_y[start_index_in_batch:end_index_in_batch] = labels_of_class_i[
                                                                       start_indices[i]:end_indices[i]]
                    start_index_in_batch = end_index_in_batch

            # shuffle images inside batch because they're ordered by label
            perm = np.arange(batch_size)

            Dataset.rg.shuffle(perm)
            batch_x = batch_x[perm]
            batch_y = batch_y[perm]

        return batch_x, batch_y

    def reset_util_fields(self):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._indices_in_epoch = np.zeros(self.num_classes, np.int32)
        self._indices_per_class = np.arange(self.num_classes) == np.argmax(self.labels, axis=1)[:, np.newaxis]
        self._counts_per_class = np.sum(self._indices_per_class, axis=0)
        self._label_distr = self._counts_per_class / np.sum(self._counts_per_class)

    def impose_distribution(self, num_examples, weights):
        train_subset_x, train_subset_y = self.next_batch(batch_size=num_examples, weights=weights)

        # overwrite current images and labels and their number
        self._num_examples = train_subset_x.shape[0]
        self._images = train_subset_x
        self._labels = train_subset_y

        # reset all the others field
        self.reset_util_fields()

    def oversampling_wrt_distribution(self, weights):
        """
            Oversampling the set in order to finally satisfy the distribution represented by weights parameter.
        :param weights: target distribution
        """

        current_distr = self.label_distr
        target_distr = weights
        current_distr[current_distr == 0] = 1  # for preventing divide by zero
        ratio = (target_distr / current_distr) / np.min((target_distr / current_distr))
        target_counts = np.ceil(self.counts_per_class * ratio).astype(np.int32)

        self._oversampling_until_counts(target_counts)

    def oversampling_until_exceed_min_count(self, target_min_count):
        """
            Oversampling the dataset in order to finally have more that min_count images in every class.
            Keep the same label distribution.
        :param target_min_count: the threshold need to be exceeded by oversampling
        """

        # exclude the labels which counts are 0, for preventing divide by 0
        current_min_count = np.min(self.counts_per_class[self.counts_per_class > 0])
        if current_min_count < target_min_count:
            ratio = target_min_count / current_min_count
            target_counts = np.ceil(self.counts_per_class * ratio).astype(np.int32)

            self._oversampling_until_counts(target_counts)

    def _oversampling_until_counts(self, target_counts):
        new_indices = np.empty(np.sum(target_counts), dtype=np.int32)
        pos = 0
        for i_class in range(self.num_classes):
            current_count_i_class = self.counts_per_class[i_class]
            if current_count_i_class == 0:
                continue
            new_class_i_indices = np.empty(target_counts[i_class], dtype=np.int32)
            indices_of_class_i = np.where(self.indices_per_class[:, i_class])[0]
            k = 0
            while k < target_counts[i_class]:
                if target_counts[i_class] > (k + current_count_i_class):
                    count_to_add = current_count_i_class
                else:
                    count_to_add = target_counts[i_class] - k
                new_class_i_indices[k:k + count_to_add] = indices_of_class_i[0:count_to_add]
                k += count_to_add
            new_indices[pos:pos + target_counts[i_class]] = new_class_i_indices
            pos += target_counts[i_class]

        # overwrite current images and labels and their number
        self._images = self.images[new_indices]
        self._labels = self.labels[new_indices]
        self._num_examples = self._images.shape[0]

        # reset all the others field
        self.reset_util_fields()

    def enhance_with_random_rotate(self, ratio=1):
        enh_train_images, enh_train_labels = enhance_with_random_rotate(self.images, np.argmax(self.labels, axis=1),
                                                                        ratio)
        # overwrite current images and labels and their number
        self._images = enh_train_images
        self._labels = utils.dense_to_one_hot(enh_train_labels, MNISTDataset.num_classes)
        self._num_examples = self._images.shape[0]

        # reset all the others field
        self.reset_util_fields()

    def enhance_with_random_zoomin(self, ratio=1):
        enh_train_images, enh_train_labels = enhance_with_random_zoomin(self.images, np.argmax(self.labels, axis=1),
                                                                        ratio)
        # overwrite current images and labels and their number
        self._images = enh_train_images
        self._labels = utils.dense_to_one_hot(enh_train_labels, MNISTDataset.num_classes)
        self._num_examples = self._images.shape[0]

        # reset all the others field
        self.reset_util_fields()

    def enhance_with_random_zoomin_and_rotate(self, ratio=1):
        enh_train_images, enh_train_labels = enhance_with_random_zoomin_and_rotate(self.images,
                                                                                   np.argmax(self.labels, axis=1),
                                                                                   ratio)
        # overwrite current images and labels and their number
        self._images = enh_train_images
        self._labels = utils.dense_to_one_hot(enh_train_labels, MNISTDataset.num_classes)
        self._num_examples = self._images.shape[0]

        # reset all the others field
        self.reset_util_fields()


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
        train_labels = utils.dense_to_one_hot(train_labels, MNISTDataset.num_classes)
        validation_labels = utils.dense_to_one_hot(validation_labels, MNISTDataset.num_classes)
        test_labels = utils.dense_to_one_hot(test_labels, MNISTDataset.num_classes)

        self._train = Dataset(train_images, train_labels, MNISTDataset.num_classes)
        self._validation = Dataset(validation_images, validation_labels, MNISTDataset.num_classes)
        self._test = Dataset(test_images, test_labels, MNISTDataset.num_classes)

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

    def impose_distribution(self, weights, global_max_weight=None, max_training_size=None, max_test_size=None):
        """
        Overwrites current MNIST dataset with a subset w.r.t. a given distribution

        :param weights: label distribution
        :param global_max_weight: - if multiple distributions will be considered in training, than we might need the
                                global maximum weight value in order to build subsets of the same size for
                                all distributions considered;
                                 - if it's None, than global_max_weight will be local maximum (i.e. the maximum value
                                 of the current weights considered)
         :param max_training_size: if is not None, the train subset will contain max_training_size samples (if possible)
         :param max_test_size: if is not None, the test subset will contain max_test_size samples (if possible)

        """

        if global_max_weight is None:
            max_weight = np.max(weights)
        else:
            max_weight = global_max_weight

        # restore original dataset and reset start indices in order to start building the subset from the beginning
        self.train.reset_indices_in_epoch()
        self.validation.reset_indices_in_epoch()
        self.test.reset_indices_in_epoch()

        train_num_examples = np.floor(np.min(self.train.counts_per_class) / max_weight).astype(np.int32)
        # round to hundreds
        train_num_examples -= train_num_examples % 100

        if max_training_size is not None and train_num_examples > max_training_size:
            train_num_examples = max_training_size
        self.train.impose_distribution(num_examples=train_num_examples, weights=weights)

        validation_num_examples = np.floor(np.min(self.validation.counts_per_class) / max_weight).astype(np.int32)
        # round to hundreds
        validation_num_examples -= validation_num_examples % 100
        self.validation.impose_distribution(num_examples=validation_num_examples, weights=weights)

        test_num_examples = np.floor(np.min(self.test.counts_per_class) / max_weight).astype(np.int32)
        # round to hundreds
        test_num_examples -= test_num_examples % 100
        if max_test_size is not None and test_num_examples > max_test_size:
            test_num_examples = max_test_size
        self.test.impose_distribution(num_examples=test_num_examples, weights=weights)


class CIFAR10Dataset:
    image_size = 32
    num_channels = 3  # RGB
    num_classes = 10

    train_validation_size = 50000
    train_size = 45000  # number of images and labels for train dataset
    validation_size = 5000  # number of images and labels for validation dataset
    test_size = 10000  # number of images and labels for test dataset

    def __init__(self, data_dir):
        train_validation_images = None
        train_validation_labels = None

        for i in range(1, 6):
            data_dic = CIFAR10Dataset.unpickle("{}/data_batch_{}".format(data_dir, i))
            if i == 1:

                train_validation_images = data_dic['data']
                train_validation_labels = data_dic['labels']
            else:
                train_validation_images = np.vstack((train_validation_images, data_dic['data']))
                train_validation_labels = np.append(train_validation_labels, data_dic['labels'])

        train_validation_images = train_validation_images.reshape((CIFAR10Dataset.train_validation_size, 3, 32, 32))
        train_validation_images = np.rollaxis(train_validation_images, axis=1, start=4)  # in order to get 32 x 32 x 3
        train_validation_images = train_validation_images / 255.0

        test_data_dic = CIFAR10Dataset.unpickle("{}/test_batch".format(data_dir))
        test_images = np.array(test_data_dic['data']).reshape((CIFAR10Dataset.test_size, 3, 32, 32))
        test_images = test_images / 255.0
        test_images = np.rollaxis(test_images, axis=1, start=4)
        test_labels = np.array(test_data_dic['labels'])

        train_images, validation_images, train_labels, validation_labels = train_test_split(train_validation_images,
                                                                                            train_validation_labels,
                                                                                            test_size=CIFAR10Dataset.validation_size,
                                                                                            random_state=42)

        # convert labels into one-hot vectors
        train_labels = utils.dense_to_one_hot(train_labels, MNISTDataset.num_classes)
        validation_labels = utils.dense_to_one_hot(validation_labels, MNISTDataset.num_classes)
        test_labels = utils.dense_to_one_hot(test_labels, MNISTDataset.num_classes)

        # # save a copy of original dataset
        # self._train_images_copy = np.copy(train_images)
        # self._train_labels_copy = np.copy(train_labels)
        # self._validation_images_copy = np.copy(validation_images)
        # self._validation_labels_copy = np.copy(validation_labels)
        # self._test_images_copy = np.copy(test_images)
        # self._test_labels_copy = np.copy(test_labels)

        self._train = Dataset(train_images, train_labels, MNISTDataset.num_classes)
        self._validation = Dataset(validation_images, validation_labels, MNISTDataset.num_classes)
        self._test = Dataset(test_images, test_labels, MNISTDataset.num_classes)

        # reset Dataset's random generator
        Dataset.reset_rg()

    def my_train_test_split(self, train_validation_images, train_validation_labels):
        perm = np.arange(0, CIFAR10Dataset.train_validation_size)
        Dataset.rg.shuffle(perm)
        train_images = train_validation_images[perm[:CIFAR10Dataset.train_size]].copy()
        train_labels = train_validation_labels[perm[:CIFAR10Dataset.train_size]].copy()
        validation_images = train_validation_images[perm[CIFAR10Dataset.train_size:]].copy()
        validation_labels = train_validation_labels[perm[CIFAR10Dataset.train_size:]].copy()
        del train_validation_images
        del train_validation_labels
        return train_images, validation_images, train_labels, validation_labels

    def restore_original_dataset(self):
        self._train = Dataset(np.copy(self._train_images_copy), np.copy(self._train_labels_copy),
                              MNISTDataset.num_classes)
        self._validation = Dataset(np.copy(self._validation_images_copy), np.copy(self._validation_labels_copy),
                                   MNISTDataset.num_classes)
        self._test = Dataset(np.copy(self._test_images_copy), np.copy(self._test_labels_copy), MNISTDataset.num_classes)

        # reset Dataset's random generator
        Dataset.reset_rg()

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            return pickle.load(fo, encoding='latin-1')

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test

    def impose_distribution(self, weights, global_max_weight=None, max_training_size=None, max_test_size=None):
        """
        Overwrites current CIFAR dataset with a subset w.r.t. a given distribution

        :param weights: label distribution
        :param global_max_weight: - if multiple distributions will be considered in training, than we might need the
                                global maximum weight value in order to build subsets of the same size for
                                all distributions considered

                                 - if it's None, than global_max_weight will be local maximum (i.e. the maximum value
                                 of the current weights considered)
         :param max_training_size: if is not None, the train subset will contain max_training_size samples (if possible)
         :param max_test_size: if is not None, the test subset will contain max_test_size samples (if possible)

        """

        if global_max_weight is None:
            max_weight = np.max(weights)
        else:
            max_weight = global_max_weight

        # restore original dataset and reset start indices in order to start building the subset from the beginning
        self.train.reset_indices_in_epoch()
        self.validation.reset_indices_in_epoch()
        self.test.reset_indices_in_epoch()

        train_num_examples = np.floor(np.min(self.train.counts_per_class) / max_weight).astype(np.int32)
        # round to hundreds
        train_num_examples -= train_num_examples % 100

        if max_training_size is not None and train_num_examples > max_training_size:
            train_num_examples = max_training_size
        self.train.impose_distribution(num_examples=train_num_examples, weights=weights)

        validation_num_examples = np.floor(np.min(self.validation.counts_per_class) / max_weight).astype(np.int32)
        # round to hundreds
        validation_num_examples -= validation_num_examples % 100
        self.validation.impose_distribution(num_examples=validation_num_examples, weights=weights)

        test_num_examples = np.floor(np.min(self.test.counts_per_class) / max_weight).astype(np.int32)
        # round to hundreds
        test_num_examples -= test_num_examples % 100
        if max_test_size is not None and test_num_examples > max_test_size:
            test_num_examples = max_test_size
        self.test.impose_distribution(num_examples=test_num_examples, weights=weights)

    @property
    def summary(self):
        return """
        training data set: images = {}, labels = {}, distr = {}
        validation data set: images = {}, labels = {}, distr = {}
        testing data set: images = {}, labels = {}, distr = {}
        """.format(self._train.images.shape, self._train.labels.shape,
                   np.round(self._train.label_distr, decimals=3),
                   self._validation.images.shape, self._validation.labels.shape,
                   np.round(self._validation.label_distr, decimals=3),
                   self._test.images.shape, self._test.labels.shape,
                   np.round(self._test.label_distr, decimals=3))
