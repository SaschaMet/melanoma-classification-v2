import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from hyperparameters.get_hyperparameter import get_hyperparameter

rng = tf.random.Generator.from_seed(42, alg='philox')


def randomCutout(x, seed=42, img_size=None):
    if img_size is None:
        img_size = get_hyperparameter("IMG_SHAPE")
    if len(x.shape) != 4:
        x = tf.expand_dims(x, 0)
    box_size = tf.cast(tf.math.floor(img_size / 10) * 2, tf.int32)
    x = tfa.image.random_cutout(
        x, (box_size, box_size), constant_values=3, seed=seed)
    return x[0]


def image_warp(x, img_size=None):
    if img_size is None:
        img_size = get_hyperparameter("IMG_SHAPE")
    if len(x.shape) != 4:
        x = tf.expand_dims(x, 0)
    flow_shape = [1, img_size, img_size, 2]
    init_flows = np.float32(np.random.normal(size=flow_shape) * 1.0)
    x = tf.cast(x, tf.float32)
    x = tf.image.resize_with_crop_or_pad(x, img_size, img_size)
    x = tfa.image.dense_image_warp(x, init_flows)
    return x[0]


def mean_filter2d(x):
    if len(x.shape) != 4:
        x = tf.expand_dims(x, 0)
    x = tfa.image.mean_filter2d(x, filter_shape=5)
    return x[0]


def random_rotate(x):
    if len(x.shape) != 4:
        x = tf.expand_dims(x, 0)
    x = tfa.image.rotate(x, tf.random.uniform(shape=(), minval=0, maxval=1))
    return x[0]


def random_transform(x, img_size=None):
    if img_size is None:
        img_size = get_hyperparameter("IMG_SHAPE")
    if len(x.shape) != 4:
        x = tf.expand_dims(x, 0)
    dx = tf.random.uniform(shape=(), minval=-int(img_size / 10),
                           maxval=int(img_size / 10), dtype=tf.int32)
    dy = tf.random.uniform(shape=(), minval=-int(img_size / 10),
                           maxval=int(img_size / 10), dtype=tf.int32)
    x = tfa.image.translate(x, translations=[dx, dy])
    return x[0]


def gaussian_filter(x):
    if len(x.shape) != 4:
        x = tf.expand_dims(x, 0)
    x = tfa.image.gaussian_filter2d(x, filter_shape=(3, 3), sigma=(1, 1))
    return x[0]


def random_flip(image, seed):
    image = tf.image.stateless_random_flip_up_down(image, seed)
    image = tf.image.stateless_random_flip_left_right(image, seed)
    return image


def random_jpeg_quality(image, seed, quality_delta=[70, 100]):
    image = tf.image.stateless_random_jpeg_quality(
        image, quality_delta[0], quality_delta[1], seed)
    return image


def random_saturation(image, seed, saturation_delta=[0.5, 1]):
    image = tf.image.stateless_random_saturation(
        image, saturation_delta[0], saturation_delta[1], seed)
    return image


def random_hue(image, seed, hue_delta=[0, 0.2]):
    hue = np.random.uniform(hue_delta[0], hue_delta[1])
    image = tf.image.stateless_random_hue(image, hue, seed)
    return image


def random_crop(image, seed, img_size=None, crop_factor=0.9):
    if img_size is None:
        img_size = get_hyperparameter("IMG_SHAPE")
    tf.image.stateless_random_crop(
        image,
        size=[tf.cast(img_size * crop_factor, tf.int32),
              tf.cast(img_size * crop_factor, tf.int32), 3],
        seed=seed)
    return image


def random_contrast(image, seed):
    image = tf.image.stateless_random_contrast(
        image, lower=0.5, upper=1.2, seed=seed)
    return image


def random_brightness(image, seed, brightness_delta=[0.1, 0.2]):
    brightness = np.random.uniform(brightness_delta[0], brightness_delta[1])
    image = tf.image.stateless_random_brightness(
        image, max_delta=brightness, seed=seed)
    return image


def augment(image_batch, seed):
    image, label = image_batch

    # Make a new seed.
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    # Apply augmentations
    image = random_flip(image, seed=new_seed)
    image = random_rotate(image)
    image = random_crop(image, seed=new_seed)
    image = random_transform(image)
    image = randomCutout(image)

    # aug_func = np.random.choice([
    #     random_jpeg_quality,
    #     random_saturation,
    #     random_brightness,
    #     random_contrast,
    #     random_hue,
    # ])
    # image = aug_func(image, seed=new_seed)

    # aug_func = np.random.choice([
    #     image_warp,
    #     mean_filter2d,
    #     gaussian_filter
    # ])
    # image = aug_func(image)

    return image, label


def augment_image(x, y):
    """Create a wrapper function for updating seeds.

    Args:
        x (Array): Image
        y (Int): Label

    Returns:
        Array: Tuple of image and label
    """
    seed = rng.make_seeds(2)[0]
    image, label = augment((x, y), seed)
    return image, label
