import os
from datetime import datetime

import numpy as np
import tensorflow as tf


class Utils(object):

    @staticmethod
    def concat_images(images, image_size, num_images_on_x, num_images_on_y):
        big_image = np.zeros((num_images_on_x * image_size, num_images_on_y * image_size))
        for i in range(num_images_on_x):
            for j in range(num_images_on_y):
                if i * num_images_on_y + j < images.shape[0]:
                    big_image[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size] = \
                        images[i * num_images_on_y + j, :, :, 0]
        return big_image

    @staticmethod
    def now_as_str():
        return "{:%Y_%m_%d---%H_%M}".format(datetime.now())

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        return (np.arange(num_classes) == labels_dense[:, np.newaxis]).astype(np.float32)

    @staticmethod
    def accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

    @staticmethod
    def plot_acc_matrix(train_distributions, acc_matrix, distr_matrix=None, use_percent_for_accuracies=False,
                        std_matrix=None, title=None):
        """
        A function which builds the plot of the accuracies obtained from multiple tests similar to a confusion matrix.
         First, a model is considered and a set of distributions. Then that model is trained on each distribution
         and tested on all distributions considered, so we end with a matrix of accuracies. The aim is to illustrate
         the issue of prior distribution shift.

        :param train_distributions: the set of distributions considered in training
        :param acc_matrix: the accuracies obtained by training and testing a model as was described above
        :param distr_matrix: the wrong predicted/wrong actual/correct actual distribution matrix could be placed
        :param use_percent_for_accuracies: if True, accuracy is printed in percents
        :param std_matrix: standard deviation matrix, for the case when acc_matrix is an average
        :param title: main title of figure
        :return: the resulted plot
        """
        from matplotlib import pyplot as plt
        from PIL import Image
        from matplotlib.image import BboxImage
        from matplotlib.transforms import Bbox, TransformedBbox

        tick_marks = np.array(range(len(train_distributions))) + 0.5
        np.set_printoptions(precision=3)
        main_fig = plt.figure(figsize=(40, 30), dpi=100)
        main_ax = plt.subplot(1, 1, 1, facecolor='w')
        ind_array = np.arange(len(train_distributions))
        x, y = np.meshgrid(ind_array, ind_array)
        temp_fig = plt.figure()
        temp_fig.facecolor = 1
        temp_ax = temp_fig.add_subplot(1, 1, 1)
        im = main_ax.imshow(acc_matrix, interpolation='nearest', cmap='Greys')
        main_ax.set_yticks(tick_marks, minor=True)
        main_ax.set_xticks(tick_marks, minor=True)
        main_ax.xaxis.set_ticks_position('none')
        main_ax.yaxis.set_ticks_position('none')
        main_ax.grid(True, which='minor', linestyle='-')
        main_fig.subplots_adjust(bottom=0.15)
        if title is not None:
            main_ax.set_title(title, y=1.15, fontsize=75)
        main_fig.colorbar(im)
        xlocations = np.array(range(len(train_distributions)))
        main_ax.set_xticks(xlocations)
        main_ax.set_xticklabels(range(len(train_distributions)))
        main_ax.set_yticks(xlocations)
        main_ax.set_yticklabels(range(len(train_distributions)))
        main_ax.xaxis.set_ticks_position('top')
        main_ax.tick_params(labelsize=15)
        main_ax.set_xlabel('Test Distribution', fontsize=50)
        main_ax.set_ylabel('Train Distribution', fontsize=50)
        main_ax.yaxis.set_label_position('right')
        temp_ax.set_frame_on(False)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = acc_matrix[y_val][x_val]
            if std_matrix is not None:
                d = std_matrix[y_val][x_val]
            if distr_matrix is None:
                vertical_offset_acc_text = 0.0
                font_size_ref = 14
            else:
                vertical_offset_acc_text = -0.4
                font_size_ref = 10
            if use_percent_for_accuracies:
                if std_matrix is None:
                    text_to_print = "{:0.1f}%".format(c * 100)
                else:
                    text_to_print = "{:0.1f}%\n\u00B1{:0.1f}%".format(c * 100, d * 100)
            else:
                if std_matrix is None:
                    text_to_print = "{:0.3f}".format(c)
                else:
                    text_to_print = "{:0.3f}\n\u00B1{:0.3f}".format(c, d)

            main_ax.text(x_val, y_val + vertical_offset_acc_text, text_to_print, color='red',
                         fontsize=font_size_ref * 21 / len(train_distributions), va='center', ha='center')
            if distr_matrix is not None:
                current_distr = distr_matrix[y_val][x_val]
                # current_norm_distr = current_distr / (np.sum(current_distr))

                # plot the current distribution on the temporary figure
                temp_ax.bar(range(10), current_distr)
                # temp_ax.set_ylim(bottom=0, top=np.max(distr_matrix[y_val, :]))
                temp_ax.set_xticks(range(10))
                temp_ax.set_xticklabels(range(10), fontsize=20)
                temp_ax.tick_params(labelsize=20)

                bbox_axis = Bbox.from_bounds(x_val / len(train_distributions) + 0.01,
                                             (len(train_distributions) - 1 - y_val) / len(train_distributions),
                                             1.0 / len(train_distributions),
                                             1.0 / len(train_distributions))
                bbox_axis = TransformedBbox(bbox_axis, main_ax.transAxes)
                bbox_image = BboxImage(bbox_axis, norm=None, origin=None, clip_on=False, zorder=1)

                # draw the renderer
                temp_fig.canvas.draw()

                # Get the RGB buffer from the temporary figure and build an image using it
                w, h = temp_fig.canvas.get_width_height()
                buf = np.frombuffer(temp_fig.canvas.tostring_argb(), dtype=np.uint8).reshape(w, h, 4)
                buf_copy = buf.copy()
                temp = buf_copy[:, :, 0].copy()
                buf_copy[:, :, 0:-1] = buf_copy[:, :, 1:]
                buf_copy[:, :, 3] = temp

                # make white pixels transparent
                pos_white_pixels = np.logical_and(np.logical_and(buf_copy[:, :, 0] == 255, buf_copy[:, :, 1] == 255),
                                                  buf_copy[:, :, 2] == 255)
                buf_copy[pos_white_pixels, 3] = 0

                distr_plot_image = Image.frombytes('RGBA', buf.shape[0:2], buf_copy)

                # Populate the box image with the current distribution plot
                bbox_image.set_data(distr_plot_image)

                # Add it to the main figure
                main_ax.add_artist(bbox_image)

                # clear temporary figure
                temp_ax.clear()

        temp_ax.set_frame_on(True)
        for idx, distr in enumerate(train_distributions):
            # plot the current distribution on the temporary figure
            temp_ax.bar(range(10), distr)
            temp_ax.set_xticks(range(10))
            temp_ax.set_xticklabels(range(10), fontsize=20)
            temp_ax.tick_params(labelsize=20)

            # build 2 box images (one for x_axis, one for y_axis) which will be populated with the current distr. plot
            bbox_x_axis = Bbox.from_bounds(1.0 * idx / len(train_distributions),
                                           1.015,
                                           1.0 / len(train_distributions),
                                           1.0 / len(train_distributions))
            bbox_x_axis = TransformedBbox(bbox_x_axis, main_ax.transAxes)
            bbox_image_x_axis = BboxImage(bbox_x_axis, norm=None, origin=None, clip_on=False)

            bbox_y_axis = Bbox.from_bounds(- 1.3 / len(train_distributions),
                                           1.0 * (len(train_distributions) - idx - 1) / len(train_distributions),
                                           1.0 / len(train_distributions),
                                           1.0 / len(train_distributions))
            bbox_y_axis = TransformedBbox(bbox_y_axis, main_ax.transAxes)
            bbox_image_y_axis = BboxImage(bbox_y_axis, norm=None, origin=None, clip_on=False)

            # draw the renderer
            temp_fig.canvas.draw()

            # Get the RGB buffer from the temporary figure and build an image using it
            w, h = temp_fig.canvas.get_width_height()
            buf = np.frombuffer(temp_fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(w, h, 3)
            distr_plot_image = Image.frombytes('RGB', buf.shape[0:2], buf)

            # Populate the box image with the current distribution plot
            bbox_image_x_axis.set_data(distr_plot_image)
            bbox_image_y_axis.set_data(distr_plot_image)

            # Add it to the main figure
            main_ax.add_artist(bbox_image_x_axis)
            main_ax.add_artist(bbox_image_y_axis)

            # clear temporary figure
            temp_ax.clear()

        # close temporary figure
        plt.close(temp_fig)

        return plt

    @staticmethod
    def restore_variable_from_checkpoint(ckpt_dir, ckpt_file, var_name):
        reader = tf.train.NewCheckpointReader(os.path.join(ckpt_dir, ckpt_file))
        if reader.has_tensor(var_name):
            return reader.get_tensor(var_name)
        print('Variable {} not found in {}{}\n'.format(var_name, ckpt_dir, ckpt_file))
        return None

    @staticmethod
    def get_all_files_from_dir_ending_with(directory, ending, without_file_extension=False):
        file_list = []
        files = os.listdir(directory)
        files.sort(key=lambda fn: os.path.getmtime(os.path.join(directory, fn)))  # sort by date
        for file in files:
            if file.endswith(ending):
                if without_file_extension:
                    file_list.append(os.path.splitext(file)[0])
                else:
                    file_list.append(file)
        return file_list

    @staticmethod
    def get_indices_wrt_distr(labels, weights, global_max_weight=None, max_no_examples=None):
        from dataset import Dataset
        """
          :param labels: list of labels for imposing distribution
          :param weights: label distribution
          :param global_max_weight: - if multiple distributions will be considered in training, than we might need the
                                  global maximum weight value in order to build subsets of the same size for
                                  all distributions considered, so we need to take it into account when building the
                                  subset
                                   - if it's None, than global_max_weight will be local maximum (i.e. the maximum value
                                   of the current weights considered)
          :param max_no_examples: if is not None, the subset will contain only max_no_examples samples (if possible)
          """

        if global_max_weight is None:
            max_weight = np.max(weights)
        else:
            max_weight = global_max_weight

        counts_per_class = np.bincount(labels, minlength=10)
        no_examples = np.floor(np.min(counts_per_class) / max_weight).astype(np.int32)
        if max_no_examples is not None and no_examples > max_no_examples:
            no_examples = max_no_examples

        # make sure that weights sum to 1
        weights = np.array(weights)
        weights = weights / sum(weights)
        # print('weights = ', weights)
        num_examples_from_each_class = np.floor(weights * no_examples).astype(np.int32)
        # print('num_examples_from_each_class = ', num_examples_from_each_class)
        # if we don't have already no_samples examples, share the remaining ones, starting with the most weighted class
        diff = no_examples - np.sum(num_examples_from_each_class)
        # print('diff = ', diff)
        if diff > 0:
            indices_sorted_weights = np.argsort(-weights)  # sort descending
            k = 0
            while diff > 0:
                num_examples_from_each_class[indices_sorted_weights[k]] += 1
                diff -= 1
                k = (k + 1) % len(weights)

        indices_wrt_distr = None
        # k = 0
        # while sum(num_examples_from_each_class) > 0:
        #     if num_examples_from_each_class[labels[k]] > 0:
        #         indices_wrt_distr.append(k)
        #         num_examples_from_each_class[labels[k]] -= 1
        #     k += 1

        indices_per_class = (np.arange(10) == labels[:, np.newaxis])
        for i in range(10):
            if num_examples_from_each_class[i] > 0:
                indices_of_class_i = np.where(indices_per_class[:, i] == True)[0]
                # print(indices_of_class_i[0:num_examples_from_each_class[i]])
                # print(np.sum(indices_of_class_i[0:num_examples_from_each_class[i]]))
                random_indices_to_append = indices_of_class_i[
                    Dataset.rg.randint(0, counts_per_class[i], num_examples_from_each_class[i])]
                if indices_wrt_distr is None:
                    indices_wrt_distr = random_indices_to_append
                else:
                    indices_wrt_distr = np.append(indices_wrt_distr, random_indices_to_append)

        return indices_wrt_distr
