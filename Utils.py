import tensorflow as tf


def load(image_file, reverse=0):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]

    w = w // 3
    input_image = image[:, :w, :]
    real_seg_image = image[:, w:w*2, :]
    target_seg_image = image[:, w*2:, :]
    input_image = tf.cast(input_image, tf.float32)
    real_seg_image = tf.cast(real_seg_image, tf.float32)
    target_seg_image = tf.cast(target_seg_image, tf.float32)
    return input_image, real_seg_image, target_seg_image


def resize(input_image, real_seg_image, target_seg_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_seg_image = tf.image.resize(real_seg_image, [height, width],
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_seg_image = tf.image.resize(target_seg_image, [height, width],
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_seg_image, target_seg_image


def random_crop(input_image, real_seg_image, target_seg_image, shape):
    stacked_image = tf.stack(
        [input_image, real_seg_image, target_seg_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[3, shape[0], shape[1], 3])

    return cropped_image[0], cropped_image[1], cropped_image[2]


def normalize(input_image, real_seg_image, target_seg_image):
    input_image = (input_image / 127.5) - 1
    real_seg_image = (real_seg_image / 127.5) - 1
    target_seg_image = (target_seg_image / 127.5) - 1
    return input_image, real_seg_image, target_seg_image


def random_mirror(input_image, real_seg_image, target_seg_image):
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_seg_image = tf.image.flip_left_right(real_seg_image)
        target_seg_image = tf.image.flip_left_right(target_seg_image)
        return input_image, real_seg_image, target_seg_image
    else:
        return input_image, real_seg_image, target_seg_image


@tf.function()
def random_jitter(input_image, real_seg_image, target_seg_image, shape):

    input_image, real_seg_image, target_seg_image = resize(
        input_image, real_seg_image, target_seg_image, shape[0], shape[1])
    input_image, real_seg_image, target_seg_image = random_crop(
        input_image, real_seg_image, target_seg_image, shape)
    input_image, real_seg_image, target_seg_image = random_mirror(
        input_image, real_seg_image, target_seg_image)
    return input_image, real_seg_image, target_seg_image


def load_image_train(image_file, shape):
    input_image, real_seg_image, target_seg_image = load(image_file)
    input_image, real_seg_image, target_seg_image = random_jitter(
        input_image, real_seg_image, target_seg_image, shape)
    input_image, real_seg_image, target_seg_image = normalize(
        input_image, real_seg_image, target_seg_image)

    return input_image, real_seg_image, target_seg_image


def load_image_test(image_file, shape):
    input_image, real_seg_image, target_seg_image = load(image_file)
    input_image, real_seg_image, target_seg_image = resize(
        input_image, real_seg_image, target_seg_image, shape[0], shape[1])
    input_image, real_seg_image, target_seg_image = normalize(
        input_image, real_seg_image, target_seg_image)

    return input_image, real_seg_image, target_seg_image
