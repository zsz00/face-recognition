import tensorflow as tf

from augmentations import random_color_manipulations,\
    random_flip_left_right, random_pixel_value_scale, random_jitter_boxes,\
    random_colored_patches


SHUFFLE_BUFFER_SIZE = 2000
# this should be some value > rows in single shard
PREFETCH_BUFFER_SIZE = 100
NUM_THREADS = 8


class Pipeline:
    """Input pipeline for training or evaluating image classifiers."""

    def __init__(self, filenames, batch_size, image_size, num_gpu=1,
                 repeat=False, shuffle=False, augmentation=False):
        """
        Arguments:
            filename: a list of strings, paths to tfrecords files.
            batch_size: an integer.
            image_size: a list with two integers [width, height],
                images of this size will be in a batch.
            num_gpu: an integer.
            repeat: whether to repeat the dataset indefinitely.
            shuffle: whether to shuffle the dataset.
            augmentation: whether to do data augmentation.
        """
        self.image_width, self.image_height = image_size
        self.augmentation = augmentation

        multi_gpu = num_gpu > 1
        assert batch_size % num_gpu == 0

        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        self.num_examples = num_examples

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=num_shards)
        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=num_shards, block_length=8
        )
        dataset = dataset.prefetch(buffer_size=batch_size)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(None if repeat else 1)

        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_THREADS)
        if multi_gpu:
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        else:
            dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        self.iterator = dataset.make_one_shot_iterator()

    def get_batch(self):
        """
        Returns:
            features: a dict with the following keys
                'images': a float tensor with shape [batch_size, 3, image_height, image_width].
            labels: a dict with the following keys
                'labels': an int tensor with shape [batch_size].
        """
        images, labels = self.iterator.get_next()
        features = {'images': images}
        labels = {'labels': labels}
        return features, labels

    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.

        Returns:
            image: a float tensor with shape [3, image_height, image_width],
                an RGB image with pixel values in the range [0, 1].
            label: an int tensor with shape [].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to [0, 1] range

        # get label
        labels = tf.to_int32(parsed_features['label'])

        if self.augmentation:
            image = self._augmentation_fn(image)
        else:
            image = tf.image.resize_images(
                image, [self.image_height, self.image_width],
                method=tf.image.ResizeMethod.BILINEAR
            )

        image = tf.transpose(image, perm=[2, 0, 1])  # to NCHW format
        return image, label

    def _augmentation_fn(self, image, boxes, labels):
        # there are a lot of hyperparameters here,
        # you will need to tune them all, haha.

        image, boxes, labels = random_image_crop(
            image, boxes, labels, probability=0.8,
            min_object_covered=0.0,
            aspect_ratio_range=(0.85, 1.15),
            area_range=(0.333, 0.8),
            overlap_thresh=0.3
        )
        image = tf.image.resize_images(
            image, [self.image_height, self.image_width],
            method=tf.image.ResizeMethod.BILINEAR
        )
        # if you do color augmentations before resizing, it will be very slow!

        image = random_color_manipulations(image, probability=0.7, grayscale_probability=0.07)
        image = random_pixel_value_scale(image, minval=0.85, maxval=1.15, probability=0.7)
        image = random_colored_patches(image, max_patches=20, probability=0.5, size_to_image_ratio=0.1)
        image = random_flip_left_right(image)
        return image


def get_num_samples(filename):
    return sum(1 for _ in tf.python_io.tf_record_iterator(filename))
