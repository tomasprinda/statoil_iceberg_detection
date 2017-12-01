import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets.inception_resnet_v2 import inception_resnet_v2,  inception_resnet_v2_arg_scope


from statoil import cfg
from statoil.example_generator import ExampleGenerator
from statoil.models.model import Model
from statoil.models.simple_cnn import SimpleCNN
from statoil.project_utils import dataset2arrays
from statoil.utils import pickle_dump, csv_dump, batches, shuffle_pairs, unbatch, pickle_load
# from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope


class CNNTL(Model):

    def __init__(self, conf, learning_rate, hidden_layer_size, filter_size, batch_size, epochs, trained_on):
        super().__init__(conf, trained_on)
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.filter_size = filter_size
        self.batch_size = batch_size
        self.save_filename = self.trained_on + "_model.ckpt"
        self.save_scaler_filename = self.trained_on + "_scaler.pkl"
        self.scaler = StandardScaler(copy=False)
        self.epochs = epochs



        self.init_graph()

    def init_graph(self):
        batch_size = None

        # Input
        self.tf_x = tf.placeholder(tf.float32, shape=(batch_size, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.NUM_CHANNELS), name="tf_x")
        self.tf_label = tf.placeholder(tf.float32, shape=(batch_size, cfg.NUM_LABELS), name="tf_label")
        self.is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

        X = tf.placeholder(tf.float32, shape=[None, cfg.IMG_SIZE_TL, cfg.IMG_SIZE_TL, cfg.NUM_CHANNELS_TL])

        with tf.variable_scope("basenet"):
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(
                    X, num_classes=1001, is_training=False, create_aux_logits=False)

        with tf.name_scope('basenet_feature_vector'):
            activations = tf.reshape(end_points["global_pool"], shape=(-1, None))

        with tf.variable_scope("topnet"):
            shape = activations.get_shape().as_list()
            activations_size = shape[1] * shape[2] * shape[3]
            activations = tf.reshape(activations, [-1, activations_size])  # Unroll

            fcnn1_w = tf.Variable(tf.truncated_normal([activations_size, self.hidden_layer_size], stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32)
            fcnn1_b = tf.Variable(tf.constant(0.0, shape=[self.hidden_layer_size], dtype=tf.float32), dtype=tf.float32)
            activations = tf.nn.relu(tf.matmul(activations, fcnn1_w) + fcnn1_b)
            activations = tf.layers.dropout(activations, training=self.is_training)

            fcnn2_w = tf.Variable(tf.truncated_normal([self.hidden_layer_size, cfg.NUM_LABELS], stddev=0.1, dtype=tf.float32),
                                  dtype=tf.float32)
            fcnn2_b = tf.Variable(tf.constant(0.0, shape=[cfg.NUM_LABELS], dtype=tf.float32), dtype=tf.float32)
            logits = tf.matmul(activations, fcnn2_w) + fcnn2_b
            self.p = tf.nn.softmax(logits)
            self.p = smooth_p(self.p)

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tf_label, logits=logits))

        # Optimizer
        with tf.variable_scope("optimizer"):

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.trained_on == "gentrain":
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                elif self.trained_on == "train":
                    var_topnet = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='topnet')
                    self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=var_topnet)
                else:
                    raise Exception("Wrong train_on")

        # Saver
        self.saver = tf.train.Saver()

    def train_and_save(self, trainset, devset=None, save_dir=None):
        dev_loss, dev_xs, dev_labels = None, None, None

        training_file = save_dir + self.trained_on + "_training.csv"
        csv_dump([("epoch", "train_loss", "dev_loss", "star")], training_file)

        self.scaler = pickle_load(save_dir + "gentrain_scaler.pkl")

        train_xs, train_labels = dataset2arrays(trainset, transfer_learning=True)
        self.scaler.transform(np.reshape(train_xs, (-1, 1)))

        if devset is not None:
            dev_xs, dev_labels = dataset2arrays(devset,  transfer_learning=True)
            self.scaler.transform(np.reshape(dev_xs, (-1, 1)))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run(session=sess)
            # self.saver.restore(sess, save_dir + "gentrain_model.ckpt")
            slim.assign_from_checkpoint_fn(
                cfg.BASE_DIR + "models/inception_resnet_v2_2016_08_30.ckpt",
                slim.get_model_variables('basenet')
            )(sess)

            min_loss = 100
            for epoch in range(self.epochs):
                train_loss = self._run_epoch(sess, train_xs, train_labels, is_training=True)

                if epoch % 1 == 0:
                    dev_loss = -1
                    star = " "
                    if devset is not None:
                        dev_loss = self._run_epoch(sess, dev_xs, dev_labels, is_training=False)

                        if save_dir is not None and dev_loss < min_loss:
                            min_loss = dev_loss
                            star = "*"
                            self.saver.save(sess, save_dir + self.save_filename)

                    print("Epoch: {:04d}, train_loss: {:.3f}, dev_loss: {:.3f}{}".format(epoch, train_loss, dev_loss, star))
                    csv_dump([(epoch, train_loss, dev_loss, star)], training_file, append=True)

            if devset is None:
                self.saver.save(sess, save_dir + self.save_filename)

    def predict(self, dataset, save_dir):
        self.scaler = pickle_load(save_dir + self.save_scaler_filename)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            self.saver.restore(sess, save_dir + self.save_filename)

            xs = dataset2arrays(dataset, return_labels=False, transfer_learning=True)
            self.scaler.transform(np.reshape(xs, (-1, 1)))
            batches_xs = batches(xs, self.batch_size)
            batches_p = []

            for x in batches_xs:
                feed_dict = {
                    self.tf_x: x,
                    self.is_training: False
                }
                p_val = sess.run(self.p, feed_dict=feed_dict)
                p_val = p_val[:, 0]
                batches_p.append(p_val)

        for example, p in zip(dataset, unbatch(batches_p)):
            example["p"] = p
            example["prediction"] = int(p >= 0.5)

    def _run_epoch(self, sess, xs, labels, is_training):
        loss_batches, size_batches = [], []

        if is_training:
            xs, labels = shuffle_pairs(xs, labels)

        batches_train_xs = batches(xs, self.batch_size)
        batches_train_labels = batches(labels, self.batch_size)
        for train_x, train_label in zip(batches_train_xs, batches_train_labels):
            feed_dict = {
                self.tf_x: train_x,
                self.tf_label: train_label,
                self.is_training: is_training
            }
            if is_training:
                _, loss_val = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            else:
                loss_val = sess.run(self.loss, feed_dict=feed_dict)

            loss_batches.append(loss_val)
            size_batches.append(train_x.shape[0])

        loss = np.average(loss_batches, weights=size_batches)

        return loss


def smooth_p(p, eps=1e-3):
    return tf.maximum(eps, tf.minimum(1 - eps, p))
