import random
import tensorflow as tf
from functools import partial
from sklearn.model_selection import train_test_split

from data_augmentation.augmentations import augment_image
from hyperparameters.get_hyperparameter import get_hyperparameter


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    return image


def read_tfrecord(example, labeled):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    } if labeled else {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    if labeled:
        label = tf.cast(example['target'], tf.int32)
        return image, label
    idnum = example['image_name']
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed
    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=tf.data.AUTOTUNE)
    # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.cache()  # cache ds for performance gains
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=tf.data.AUTOTUNE)

    # Resize image to given dimensions
    IMG_SHAPE = get_hyperparameter("IMG_SHAPE")
    resizing_layer = tf.keras.layers.experimental.preprocessing.Resizing(
        IMG_SHAPE, IMG_SHAPE)

    # resize the images to the same height and width
    dataset = dataset.map(lambda x, y: (resizing_layer(x), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def get_training_dataset(training_filenames, batch_size, augment=True, shuffle=True):
    dataset = load_dataset(training_filenames, labeled=True)
    REPLICAS = get_hyperparameter("REPLICAS")

    if augment:
        dataset = dataset.map(
            augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(
            128 * REPLICAS, reshuffle_each_iteration=True)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        dataset = dataset.with_options(opt)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_validation_dataset(validation_filenames, batch_size, ordered=True):
    dataset = load_dataset(validation_filenames, labeled=True, ordered=ordered)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def get_test_dataset(test_filenames, batch_size, ordered=True):
    dataset = load_dataset(test_filenames, labeled=False, ordered=ordered)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def create_datasets(GCS_PATH_2020, GCS_PATH_19_18_17):
    SEED = get_hyperparameter("SEED")
    BATCH_SIZE = get_hyperparameter("BATCH_SIZE")
    VALIDATION_SIZE = get_hyperparameter("VALIDATION_SIZE")

    training_filenames = tf.io.gfile.glob(GCS_PATH_2020 + '/train*.tfrec')
    training_filenames = training_filenames + \
        tf.io.gfile.glob(GCS_PATH_19_18_17 + '/train*.tfrec')

    test_filenames = tf.io.gfile.glob(GCS_PATH_2020 + '/test*.tfrec')

    training_filenames, validation_filenames = train_test_split(
        training_filenames, test_size=VALIDATION_SIZE, random_state=SEED)
    training_filenames = list(training_filenames)

    # Test if TRAINING and VALIDATION files are valid
    for x in training_filenames:
        if x in validation_filenames:
            raise Exception("TRAIN AND TEST FILES ARE NOT VALID!")

    random.shuffle(training_filenames)

    print("training_filenames", len(training_filenames))
    print("validation_filenames", len(validation_filenames))
    print("test_filenames", len(test_filenames))

    train_ds = get_training_dataset(training_filenames, BATCH_SIZE)
    val_ds = get_validation_dataset(validation_filenames, BATCH_SIZE)
    test_ds = get_test_dataset(test_filenames, BATCH_SIZE)

    return train_ds, val_ds, test_ds
