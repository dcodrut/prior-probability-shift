import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from laplotter import LossAccPlotter


class TrainingPlotter(object):
    def __init__(self, title, file_name, show_plot_window=False):
        self.plotter = LossAccPlotter(title=title,
                                      save_to_filepath=file_name,
                                      show_regressions=True,
                                      show_averages=True,
                                      show_loss_plot=True,
                                      show_acc_plot=True,
                                      show_plot_window=show_plot_window,
                                      x_label="Epoch")

    def add_loss_accuracy_to_plot(self, epoch, loss_train, acc_train, loss_val, acc_val, redraw=True):
        self.plotter.add_values(epoch, loss_train=loss_train, acc_train=acc_train, loss_val=loss_val, acc_val=acc_val,
                                redraw=redraw)
        return self.plotter.fig

    def safe_shut_down(self):
        self.plotter.block()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels):
        from sklearn.metrics import confusion_matrix
        cmap = plt.cm.binary
        cm = confusion_matrix(y_true, y_pred, labels=range(0, len(labels)))
        tick_marks = np.array(range(len(labels))) + 0.5
        # each line sum to the corresponding num_examples of that class => we can normalize the matrix into [0,1]
        counts_per_class = cm.sum(axis=1)
        counts_per_class[counts_per_class == 0] = 1  # in order to prevent division by zero
        cm_normalized = cm.astype('float') / counts_per_class[:, np.newaxis]
        plt.figure(figsize=(20, 16), dpi=120)
        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        intFlag = 0
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            if (intFlag):
                c = cm[y_val][x_val]
                plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=14, va='center', ha='center')
            else:
                c = cm_normalized[y_val][x_val]
                if (c > 0.001):
                    plt.text(x_val, y_val, "%0.3f" % (c,), color='red', fontsize=14, va='center', ha='center')
                else:
                    plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=14, va='center', ha='center')
        if (intFlag):
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
        else:
            plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('')
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('Index of True Classes')
        plt.xlabel('Index of Predict Classes')
        return plt

    @staticmethod
    def combine_images(images, file_name, top_images=1500):
        if len(images) > top_images:
            images = images[0:top_images - 1]
        count = len(images)
        image_size = images[0].shape[0]
        max_images_pre_row = 25
        width = max_images_pre_row * image_size
        height = math.ceil(count / max_images_pre_row) * image_size
        channels = images[0].shape[2]
        if channels == 3:
            blank_image = Image.new("RGB", (width, height))
        else:
            blank_image = Image.new("L", (width, height))

        for index in range(count):
            # each image was scale to [0,1] => we need to do the inverse operation
            image = np.array(images[index])
            if channels == 3:
                image = image[0:image_size, 0:image_size, :]
            else:
                image = image[0:image_size, 0:image_size, 0]
            image = (image * 256).astype(np.uint8)
            image = Image.fromarray(image)
            column = index % max_images_pre_row
            row = index // max_images_pre_row
            blank_image.paste(image, (column * image_size, row * image_size))
        blank_image.save(file_name)
