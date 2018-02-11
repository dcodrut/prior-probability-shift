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

