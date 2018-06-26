# Source: https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN

import numpy as np
import scipy.ndimage
import skimage.transform


def enhance_with_random_rotate(images, labels, ratio):
    return enhance_with_function(images, labels, ratio, _enhance_one_image_with_rotate_randomly)


def enhance_with_random_zoomin(images, labels, ratio):
    return enhance_with_function(images, labels, ratio, _enhance_one_image_with_zoomin_randomly)


def enhance_with_random_zoomin_and_rotate(images, labels, ratio):
    return enhance_with_function(
        images, labels, ratio,
        _enhance_one_image_with_random_funcs(
            [
                _enhance_one_image_with_rotate_randomly,
                _enhance_one_image_with_zoomin_randomly
            ]
        ))


def enhance_with_function(images, labels, ratio, enhance_func):
    inputs_per_class = np.bincount(labels)
    max_inputs = np.max(inputs_per_class)

    # One Class
    for i in range(len(inputs_per_class)):
        # input_ratio = math.ceil((max_inputs * ratio + 1 - inputs_per_class[i]) / inputs_per_class[i])
        input_ratio = ratio
        print("generating class:{} with ratio:{}, max input:{}, current:{}".format(
            i, input_ratio, max_inputs, inputs_per_class[i]))

        if input_ratio <= 1:
            continue

        new_features = []
        new_labels = []
        mask = np.where(labels == i)

        for feature in images[mask]:
            generated_images = enhance_func(feature, input_ratio)
            for generated_image in generated_images:
                new_features.append(generated_image)
                new_labels.append(i)

        images = np.append(images, new_features, axis=0)
        labels = np.append(labels, new_labels, axis=0)

    return images, labels


def _enhance_one_image_with_rotate_randomly(image, how_many_to_generate):
    _IMAGE_ROTATE_ANGLES = np.arange(-20, 20, 3)
    generated_images = []
    for index in range(how_many_to_generate):
        generated_images.append(
            scipy.ndimage.rotate(image,
                                 np.random.choice(_IMAGE_ROTATE_ANGLES),
                                 reshape=False))

    return generated_images


def _enhance_one_image_with_zoomin_randomly(image, how_many_to_generate):
    generated_images = []
    for index in range(how_many_to_generate):
        generated_image = _zoomin_image_randomly(image)
        generated_images.append(generated_image)
    return generated_images


def _zoomin_image_randomly(image):
    _IMAGE_CUT_RATIOS = np.arange(0.05, 0.2, 0.02)
    scale = np.random.choice(_IMAGE_CUT_RATIOS)
    lx, ly, _ = image.shape
    first_run = image[int(lx * scale): - int(lx * scale), int(ly * scale): - int(ly * scale), :]
    return skimage.transform.resize(first_run, (lx, ly), mode='constant')


def _enhance_one_image_with_random_funcs(enhance_funcs):
    def __f(image, how_many_to_generate):
        func_indeies = np.random.randint(0, len(enhance_funcs), size=how_many_to_generate)
        return _flatten(map(lambda i: enhance_funcs[i](image, 1), func_indeies))

    return __f


def _flatten(listoflists):
    return [item for list in listoflists for item in list]
