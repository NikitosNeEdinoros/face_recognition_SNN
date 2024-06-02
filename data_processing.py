import os
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import tensorflow_addons as tfa

PATH_POS = os.path.join('.', 'data', 'positive')
PATH_NEG = os.path.join('.', 'data', 'negative_lfw')
PATH_ANCH = os.path.join('.', 'data', 'anchor')

def preprocess(file_path):
    image = tf.io.read_file(file_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (112, 112))
    image = image / 255.0
    return image

def process(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label

def data_partition(data, train_size=0.8):
    data_size = data.cardinality().numpy()
    train_size = int(data_size * train_size)

    train = data.take(train_size)
    train = train.batch(16).prefetch(AUTOTUNE)

    test = data.skip(train_size)
    test = test.batch(16).prefetch(AUTOTUNE)

    return train, test

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.2)
    return image


def prepare_data():
    anchor = tf.data.Dataset.list_files(os.path.join(PATH_ANCH, '*.jpg')).take(300)
    negative = tf.data.Dataset.list_files(os.path.join(PATH_NEG, '*.jpg')).take(300)
    positive = tf.data.Dataset.list_files(os.path.join(PATH_POS, '*.jpg')).take(300)

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(anchor.cardinality().numpy()))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(anchor.cardinality().numpy()))))
    data = positives.concatenate(negatives)

    data = data.map(process, num_parallel_calls=AUTOTUNE)
    data = data.cache()
    data = data.shuffle(buffer_size=1024)

    def augment(anchor, positive_or_negative, label):
        anchor = augment_image(anchor)
        positive_or_negative = augment_image(positive_or_negative)
        return anchor, positive_or_negative, label

    data = data.map(augment, num_parallel_calls=AUTOTUNE)

    return data

if __name__ == '__main__':
    data = prepare_data()
    train, test = data_partition(data)

