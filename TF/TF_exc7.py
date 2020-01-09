"""
Exercise VII: For this exercise you are asked to implement a function (create_dataset) that returns a tf.data.Dataset
object holding a dataset for image classification. This dataset should read the image and its label from a CSV file
and apply some data augmentation to the images.

At the end of the script you can find some code so that you can run your pipeline and check how fast is it.
To run: `python exercise_7.py {dataset_path} {images_dir}
It will print in your terminal the time cost of generating each batch. Feel free to change to accommodate your needs.
"""
import argparse
import time
import csv
import multiprocessing
import os
import tensorflow as tf
print(tf.__version__, tf.executing_eagerly())
logdir = 'tf_sumaries'

def create_dataset(dataset_path, images_dir, num_epochs, batch_size):
    """
    :param str dataset_path: The path to the CSV file describing the dataset. Each row should be (image_name, label)
    :param str images_dir: The directory holding the images. The full image path is then the concatenation of the
    images_dir and the image_name obtained from the CSV
    :param int num_epochs: Number of epochs to generate samples
    :param int batch_size: Number of samples per batch
    :return:
    """
    dataset = tf.data.Dataset.from_generator(generator = lambda: data_generator(dataset_path, images_dir),
                                             output_types = (tf.string, tf.string),
                                             output_shapes = (tf.TensorShape([]),
                                             tf.TensorShape([])))
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.shuffle(100)
    dataset = dataset.map(create_sample)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)

    return dataset

def data_generator(path, directory):
 with open(path, 'r') as f:
 	reader = csv.reader(f)
 	for image_name, label in reader:
 		img_path = os.path.join(directory, image_name)
 		yield img_path, label

def create_sample(image_path, label):
    with tf.name_scope('create_sample'):
        with tf.name_scope('read_image'):
            raw_image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(raw_image, channels=3)

        with tf.name_scope('preprocessing'):
            mean_channel = [123.68, 116.779, 103.939]
            image = tf.cast(image, dtype=tf.float32)
            image = tf.subtract(image, mean_channel, name='mean_substraction')
            image = tf.divide(image, tf.constant(255.0, dtype = tf.float32), name ='0_1_normalization')
            image = tf.image.resize(image, size=(256, 256))

        with tf.name_scope('data_augmentation'):
            image = tf.image.random_crop(image, size=(227, 227, 3))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=20)

    return image, label



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('dataset_path', help='Path to dataset description')
    parser.add_argument('images_dir', help='Image directory')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    args = parser.parse_args()

    graph = tf.compat.v1.Graph()
    with graph.as_default():
        with tf.device('/cpu:0'):  # To force the graph operations of the input pipeline to be placed in the CPU
            with tf.name_scope('input_pipeline'):
                dataset = create_dataset(args.dataset_path, args.images_dir, args.num_epochs, args.batch_size)
                iterator = dataset.make_one_shot_iterator()
                batch = iterator.get_next()
                images, labels = batch
            with tf.device('/cpu:0'): # Here the ops should go to the GPU
                with tf.name_scope('ConvNet'):
                    x0 = tf.nn.conv2d(images, filters = tf.compat.v1.Variable(shape = [7,7,3,32], dtype = tf.float32), strides = 1, padding = 'VALID')
                    x1 = tf.nn.max_pool2d(x0, ksize = 2, strides = 1, padding = 'VALID')

    with tf.compat.v1.Session(graph=graph) as sess:
        try:
            while True:
                start = time.time()
                pred = sess.run(x1)
                #print(images[0], labels[0])
                duration = time.time() - start
                print('Time per batch: {}'.format(duration))
                #writer = tf.compat.v1.summary.FileWriter(logdir, graph = graph)
        except tf.errors.OutOfRangeError:
            pass
