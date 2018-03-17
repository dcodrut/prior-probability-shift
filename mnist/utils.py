import numpy as np
from datetime import datetime


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
        return "{:%Y_%m_%d_%H_%M}".format(datetime.now())

    @staticmethod
    def dense_to_one_hot(labels_dense, num_classes):
        """Convert class labels from scalars to one-hot vectors."""
        return (np.arange(num_classes) == labels_dense[:, np.newaxis]).astype(np.float32)

    @staticmethod
    def accuracy(predictions, labels):
        return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

    @staticmethod
    def plot_acc_matrix(train_distributions, acc_matrix):
        """
        A function which builds the plot of the accuracies obtained from multiple tests similar to a confusion matrix.
         First, a model is considered and a set of distributions. Then that model is trained on each distribution
         and tested on all distributions considered, so we end with a matrix of accuracies. The aim is to illustrate
         the issue of prior distribution shift.

        :param train_distributions: the set of distributions considered in training
        :param acc_matrix: the accuracies obtained by training and testing a model as was described above
        :return: the resulted plot
        """
        from matplotlib import pyplot as plt
        from PIL import Image
        from matplotlib.image import BboxImage
        from matplotlib.transforms import Bbox, TransformedBbox

        tick_marks = np.array(range(len(train_distributions))) + 0.5
        np.set_printoptions(precision=3)
        main_fig = plt.figure(figsize=(40, 30), dpi=100)
        main_ax = plt.subplot(1, 1, 1)
        ind_array = np.arange(len(train_distributions))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = acc_matrix[y_val][x_val]
            main_ax.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=14 * 21 / len(train_distributions),
                         va='center', ha='center')
        im = main_ax.imshow(acc_matrix, interpolation='nearest', cmap='Greys')
        main_ax.set_yticks(tick_marks, minor=True)
        main_ax.set_xticks(tick_marks, minor=True)
        main_ax.xaxis.set_ticks_position('none')
        main_ax.yaxis.set_ticks_position('none')
        main_ax.grid(True, which='minor', linestyle='-')
        main_fig.subplots_adjust(bottom=0.15)
        main_ax.set_title('')
        main_fig.colorbar(im)
        xlocations = np.array(range(len(train_distributions)))
        main_ax.set_xticks(xlocations)
        main_ax.set_xticklabels(range(len(train_distributions)))
        main_ax.set_yticks(xlocations)
        main_ax.set_yticklabels(range(len(train_distributions)))
        main_ax.xaxis.set_ticks_position('top')
        main_ax.tick_params(labelsize=15)
        main_ax.set_xlabel('Test Distribution', fontsize=30)
        main_ax.set_ylabel('Train Distribution', fontsize=30)
        main_ax.yaxis.set_label_position('right')

        temp_fig = plt.figure()
        temp_ax = temp_fig.add_subplot(1, 1, 1)
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
