import tensorflow as tf
import AntiSegmentation_GAN
import Utils

BUFFER_SIZE = 400
EPOCHS = 100
BATCH_SIZE = 1
shape = [256, 256]
training_PATH = "/data/HRSC2016/training/"
test_PATH = "/data/HRSC2016/training/"
train_dataset = tf.data.Dataset.list_files(training_PATH+'*.png')
train_dataset = train_dataset.map(
    lambda x: Utils.load_image_train(x, shape))
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(test_PATH+'*.png')
test_dataset = test_dataset.map(lambda x: Utils.load_image_train(x, shape))
test_dataset = test_dataset.batch(BATCH_SIZE)

GAN = AntiSegmentation_GAN.Antiseg()
GAN.fit(train_dataset, 20, test_dataset)
