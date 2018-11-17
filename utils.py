import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox


def concat_images(images, image_size, num_images_on_x, num_images_on_y):
    channels = images[0].shape[2]
    if channels == 1:
        big_image = np.zeros((num_images_on_x * image_size, num_images_on_y * image_size))
    else:
        big_image = np.zeros((num_images_on_x * image_size, num_images_on_y * image_size, channels))
    for i in range(num_images_on_x):
        for j in range(num_images_on_y):
            if i * num_images_on_y + j < images.shape[0]:
                start_i = i * image_size
                stop_i = (i + 1) * image_size
                start_j = j * image_size
                stop_j = (j + 1) * image_size
                if channels == 1:
                    big_image[start_i:stop_i, start_j:stop_j] = images[i * num_images_on_y + j, :, :, 0]
                else:
                    big_image[start_i:stop_i, start_j:stop_j, :] = images[i * num_images_on_y + j, :, :, :]

    return big_image


def now_as_str():
    return "{:%Y_%m_%d---%H_%M}".format(datetime.now())


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    return (np.arange(num_classes) == labels_dense[:, np.newaxis]).astype(np.float32)


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def plot_acc_matrix(train_distributions, acc_matrix, test_distributions=None, distr_matrix=None,
                    use_percent_for_accuracies=False, std_matrix=None, title=None):
    """
    A function which builds the plot of the accuracies obtained from multiple tests similar to a confusion matrix.
     First, a model is considered and a set of distributions. Then that model is trained on each distribution
     and tested on all distributions considered, so we end with a matrix of accuracies. The aim is to illustrate
     the issue of prior distribution shift.

    :param train_distributions: the set of distributions considered in training
    :param acc_matrix: the accuracies obtained by training and testing a model as was described above
    :param test_distributions: the set of distributions used for testing; if None, train_distributions are considered
    :param distr_matrix: the wrong predicted/wrong actual/correct actual distribution matrix could be placed
    :param use_percent_for_accuracies: if True, accuracy is printed in percents
    :param std_matrix: standard deviation matrix, for the case when acc_matrix is an average
    :param title: main title of figure
    :return: the resulted plot
    """
    if test_distributions is None:
        test_distributions = train_distributions

    assert (len(train_distributions) == len(test_distributions))

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
    cb = main_fig.colorbar(im)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=30)
    xlocations = np.array(range(len(train_distributions)))
    main_ax.set_xticks(xlocations)
    main_ax.set_xticklabels(range(len(train_distributions)))
    main_ax.set_yticks(xlocations)
    main_ax.set_yticklabels(range(len(train_distributions)))
    main_ax.xaxis.set_ticks_position('top')
    main_ax.tick_params(labelsize=20)
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

    for idx, train_distr in enumerate(train_distributions):
        # plot the current distribution on the temporary figure
        temp_ax.bar(range(10), train_distr)
        temp_ax.set_xticks(range(10))
        temp_ax.set_xticklabels(range(10), fontsize=20)
        temp_ax.tick_params(labelsize=20)

        # build a box images which will be populated with the current train_distr. plot
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
        bbox_image_y_axis.set_data(distr_plot_image)

        # Add it to the main figure
        main_ax.add_artist(bbox_image_y_axis)

        # clear temporary figure
        temp_ax.clear()

    for idx, test_distr in enumerate(test_distributions):
        # plot the current distribution on the temporary figure
        temp_ax.bar(range(10), test_distr)
        temp_ax.set_xticks(range(10))
        temp_ax.set_xticklabels(range(10), fontsize=20)
        temp_ax.tick_params(labelsize=20)

        # build a box images which will be populated with the current train_distr. plot
        bbox_x_axis = Bbox.from_bounds(1.0 * idx / len(test_distributions),
                                       1.015,
                                       1.0 / len(test_distributions),
                                       1.0 / len(test_distributions))
        bbox_x_axis = TransformedBbox(bbox_x_axis, main_ax.transAxes)
        bbox_image_x_axis = BboxImage(bbox_x_axis, norm=None, origin=None, clip_on=False)

        # draw the renderer
        temp_fig.canvas.draw()

        # Get the RGB buffer from the temporary figure and build an image using it
        w, h = temp_fig.canvas.get_width_height()
        buf = np.frombuffer(temp_fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(w, h, 3)
        distr_plot_image = Image.frombytes('RGB', buf.shape[0:2], buf)

        # Populate the box image with the current distribution plot
        bbox_image_x_axis.set_data(distr_plot_image)

        # Add it to the main figure
        main_ax.add_artist(bbox_image_x_axis)

        # clear temporary figure
        temp_ax.clear()

    # close temporary figure
    plt.close(temp_fig)

    return plt


def restore_variable_from_checkpoint(ckpt_dir, ckpt_file, var_name):
    reader = tf.train.NewCheckpointReader(os.path.join(ckpt_dir, ckpt_file))
    if reader.has_tensor(var_name):
        return reader.get_tensor(var_name)
    print('Variable {} not found in {}{}\n'.format(var_name, ckpt_dir, ckpt_file))
    return None


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
    num_examples_from_each_class = np.floor(weights * no_examples).astype(np.int32)

    # if we don't have already no_samples examples, share the remaining ones, starting with the most weighted class
    diff = no_examples - np.sum(num_examples_from_each_class)
    if diff > 0:
        indices_sorted_weights = np.argsort(-weights)  # sort descending
        k = 0
        while diff > 0:
            num_examples_from_each_class[indices_sorted_weights[k]] += 1
            diff -= 1
            k = (k + 1) % len(weights)

    indices_wrt_distr = None

    indices_per_class = (np.arange(10) == labels[:, np.newaxis])
    for i in range(10):
        if num_examples_from_each_class[i] > 0:
            indices_of_class_i = np.where(indices_per_class[:, i] == True)[0]
            perm = np.arange(0, counts_per_class[i])
            Dataset.rg.shuffle(perm)
            random_indices_to_append = indices_of_class_i[perm[:num_examples_from_each_class[i]]]
            if indices_wrt_distr is None:
                indices_wrt_distr = random_indices_to_append
            else:
                indices_wrt_distr = np.append(indices_wrt_distr, random_indices_to_append)

    # shuffle indices_wrt_distr because they are ordered by label
    perm = np.arange(no_examples)
    Dataset.rg.shuffle(perm)
    indices_wrt_distr = indices_wrt_distr[perm]

    return indices_wrt_distr


def distr_sequence_to_rgb_image(color_list, distr_sequence, width=800, height=150):
    lines = 1000
    cols = len(distr_sequence)
    distr_image_matrix = np.zeros((lines, cols, 3), dtype=np.uint8)
    distr_image_matrix[:, :, :] = 255
    for idc, distr in enumerate(distr_sequence):
        k = 0
        c = 0
        for idl in distr:
            distr_image_matrix[k:k + int(lines * idl), idc] = color_list[c]
            k += int(lines * idl)
            c = (c + 1) % 10
    distr_image_matrix = np.flip(distr_image_matrix, axis=0)
    pil_image = Image.fromarray(distr_image_matrix).resize((width, height))
    return pil_image


def plot_sequence_of_distr(list_no_examples_from_each_distr, distr_sequence, method, fig_title, color_list,
                           window_length=None, save_to_file=False, img_width=800):
    sequence_of_images_length = int(np.sum(list_no_examples_from_each_distr))
    if method == 'bar_plot':
        my_dpi = 100
        plt.figure(figsize=(2000 / my_dpi, 500 / my_dpi), dpi=my_dpi)
        bottom = None
        for k in range(10):
            if k == 0:
                bottom = np.zeros(len(distr_sequence))
            else:
                bottom += distr_sequence[:, k - 1]
            plt.bar(x=np.arange(sequence_of_images_length), height=distr_sequence[:, k], width=1, bottom=bottom)

        plt.tick_params(labelsize=10)
        if window_length is None:
            xticks_array = np.append([0], np.cumsum(list_no_examples_from_each_distr))
            plt.xticks(
                np.arange(0, len(distr_sequence) + 1, len(distr_sequence) / len(list_no_examples_from_each_distr)),
                xticks_array)
        else:
            plt.xticks(np.arange(0, len(distr_sequence) + 1, window_length))
        plt.yticks(np.arange(0, 1.01, 0.1))
        plt.legend(labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], fontsize=15)
        plt.ylabel('Label distribution', fontsize=20)
        plt.xlabel('Sliding window position within sequence', fontsize=20)
        plt.title(fig_title, fontsize=30, y=1.05)

    elif method == 'my_distr_img_plot':
        plt.figure(figsize=(80, 15), dpi=50)
        width = 800
        height = 150
        plt.imshow(distr_sequence_to_rgb_image(color_list, distr_sequence, width=img_width))
        plt.tick_params(labelsize=35)
        xticks_array = np.append([0], np.cumsum(list_no_examples_from_each_distr))
        if window_length is not None:
            xticks_array = np.append(xticks_array, np.cumsum(list_no_examples_from_each_distr)[:-1] + window_length)
        plt.xticks(np.arange(0, width + 1, width / len(list_no_examples_from_each_distr)), xticks_array)
        plt.yticks(np.arange(0, height, height / 10), np.round(np.arange(1, 0, -0.1), decimals=1))
        plt.ylabel('Label distribution', fontsize=50)
        plt.xlabel('Sliding window position within sequence', fontsize=50)
        plt.title(fig_title, fontsize=70, y=1.05)
    plt.grid(axis='x', linewidth=6, linestyle='--')
    if save_to_file:
        plt.savefig('{}.png'.format(fig_title), bbox_inches='tight', pad_inches=1)
    plt.show()
