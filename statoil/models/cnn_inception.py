from pprint import pprint

from statoil import cfg
from statoil.models.model import Model
from statoil.project_utils import dataset2arrays
from statoil.utils import pickle_dump, json_dump, strip_extension, pickle_load, batches, print_info, unbatch, csv_dump, shuffle_pairs, \
    variable_summaries
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow.contrib.slim as slim
from nets.inception_v4 import inception_v4
from nets.mobilenet_v1 import mobilenet_v1
from nets.resnet_v1 import resnet_v1_50


class CNNInception(Model):
    def __init__(self, conf, trained_on):
        super().__init__(conf, trained_on)
        pprint(conf)
        self.save_filename = self.trained_on + "_model.ckpt"
        self.save_scaler_filename = self.trained_on + "_scaler.pkl"
        self.scaler = StandardScaler(copy=False)

        self.global_step = 0

        self.init_graph()

    def init_graph(self):
        tf.reset_default_graph()
        batch_size = None

        # Input
        self.tf_x = tf.placeholder(tf.float32, shape=(batch_size, cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.NUM_CHANNELS), name="tf_x")
        self.tf_label = tf.placeholder(tf.float32, shape=(batch_size, cfg.NUM_LABELS), name="tf_label")
        self.is_training = tf.placeholder(tf.bool, shape=None, name="is_training")

        def dense_block(name, input_, out_size, bias, relu, dropout):
            with tf.variable_scope(name):
                in_size = input_.get_shape().as_list()[1]

                w = tf.get_variable(
                    name="w",
                    shape=[in_size, out_size],
                    # initializer=tf.truncated_normal(, stddev=0.1, dtype=tf.float32),
                    dtype=tf.float32
                )
                variable_summaries(w, w.name)

                data = tf.matmul(input_, w)
                if bias:
                    b = tf.get_variable(
                        name="b",
                        shape=[out_size],
                        # initializer=tf.constant(0.0, shape=[out_size], dtype=tf.float32), dtype=tf.float32
                    )
                    data += b
                    variable_summaries(b, b.name)
                if relu:
                    data = tf.nn.relu(data)
                if dropout:
                    data = tf.layers.dropout(data, training=self.is_training, rate=self.conf["dropout_rate"])

                # print_layer_info(name, w, b, data)
                return data

        def print_layer_info(name, w, b, data):
            w_shape = w.get_shape().as_list()
            b_shape = b.get_shape().as_list()
            data_shape = data.get_shape().as_list()[1:]
            print("{}: {} params, w:{} + b:{}".format(
                name,
                np.prod(w_shape) + np.prod(b_shape),
                "*".join([str(val) for val in w_shape]),
                "*".join([str(val) for val in b_shape])))
            print("{}: {} activations, {} per example".format(
                name,
                np.prod(data_shape),
                "*".join([str(val) for val in data_shape])))

        # Net
        # activations, endpoints = inception_v4(
        #     self.tf_x,
        #     num_classes=None,
        #     is_training=self.is_training,
        #     dropout_keep_prob=1.,  # Applies only if num_classes > 0
        #     reuse=None,
        #     scope='InceptionV4',
        #     create_aux_logits=True)

        # activations, endpoints = mobilenet_v1(
        #     self.tf_x,
        #     num_classes=None,
        #     dropout_keep_prob=0.999,
        #     is_training=self.is_training,
        #     # min_depth=self.conf["mobilenet_v1_min_depth"],
        #     # depth_multiplier=self.conf["mobilenet_v1_depth_multiplier"],
        #     min_depth=128,
        #     depth_multiplier=2,
        #     conv_defs=None,
        #     prediction_fn=None,
        #     spatial_squeeze=False,
        #     reuse=None,
        #     scope='MobilenetV1',
        #     global_pool=True)

        activations, endpoints = resnet_v1_50(
            self.tf_x,
            num_classes=None,
            is_training=self.is_training,
            global_pool=True,
            output_stride=None,
            spatial_squeeze=False,
            reuse=None,
            scope='resnet_v1_50')

        self.init_resnet = slim.assign_from_checkpoint_fn(
            cfg.RESNET_V1_WEIGHTS,
            slim.get_model_variables('resnet_v1_50')
        )

        if self.conf["inception_dropout"]:
            activations = tf.layers.dropout(activations, training=self.is_training, rate=self.conf["dropout_rate"])

        with tf.variable_scope("reshape"):
            shape = activations.get_shape().as_list()
            activations_size = shape[1] * shape[2] * shape[3]
            activations = tf.reshape(activations, [-1, activations_size])  # Unroll

        variable_summaries(activations, "after_cnn")

        # if self.conf["fcn_1"]:
        #     activations = dense_block("fcn_1", activations, self.conf["fcn_1_out_size"], bias=True, dropout=self.conf["fcn_1_dropout"], relu=True)
        # activations = dense_block("fcn_2", activations, self.conf["fcn_2_out_size"], dropout=self.conf["fcn_2_dropout"], relu=True)
        logits = dense_block("fcn_3", activations, cfg.NUM_LABELS, bias=False, dropout=False, relu=False)

        variable_summaries(logits, "logits")

        with tf.variable_scope("probability"):
            self.p = tf.nn.sigmoid(logits)
            self.p = smooth_p(self.p)

        # Loss
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tf_label, logits=logits))
            tf.summary.scalar("loss", self.loss)

        # Optimizer
        with tf.variable_scope("optimizer"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                op = tf.train.AdamOptimizer(self.conf["learning_rate"])

                grads_and_vars = op.compute_gradients(self.loss, trainables)
                for grad, var in grads_and_vars:
                    tf.summary.histogram(grad.name, grad)
                variable_summaries(grads_and_vars[0][0], "grad_" + grads_and_vars[0][1].name)
                self.optimizer = op.apply_gradients(grads_and_vars)

        # Saver
        self.saver = tf.train.Saver()
        self.merged_summary_op = tf.summary.merge_all()

    def train_and_save(self, trainset, devset=None, save_dir=None):
        dev_loss, dev_xs, dev_labels = None, None, None

        training_file = save_dir + self.trained_on + "_training.csv"
        csv_dump([("epoch", "train_loss", "dev_loss", "star")], training_file)

        train_xs, train_labels = dataset2arrays(trainset)
        self.scaler.fit_transform(np.reshape(train_xs, (-1, 1)))

        if devset is not None:
            dev_xs, dev_labels = dataset2arrays(devset)
            self.scaler.transform(np.reshape(dev_xs, (-1, 1)))

        pickle_dump(self.scaler, save_dir + self.save_scaler_filename)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run(session=sess)
            self.init_resnet(sess)

            # op to write logs to Tensorboard
            self.summary_writer_train = tf.summary.FileWriter(save_dir + "train", graph=tf.get_default_graph())
            self.summary_writer_dev = tf.summary.FileWriter(save_dir + "dev", graph=tf.get_default_graph())

            min_loss = 100
            changed_before = 0
            for epoch in range(self.conf["epochs"]):
                train_loss = self._run_epoch(sess, train_xs, train_labels, is_training=True)

                dev_loss = -1
                star = " "
                if devset is not None:
                    changed_before += 1
                    dev_loss = self._run_epoch(sess, dev_xs, dev_labels, is_training=False)

                    if save_dir is not None and dev_loss < min_loss:
                        changed_before = 0
                        min_loss = dev_loss
                        star = "*"
                        self.saver.save(sess, save_dir + self.save_filename)

                    print("Epoch: {:04d}, train_loss: {:.3f}, dev_loss: {:.3f}{}".format(epoch, train_loss, dev_loss, star))
                    csv_dump([(epoch, train_loss, dev_loss, star)], training_file, append=True)

                if 0 < self.conf["stop_on_no_change"] <= changed_before:
                    break

            if devset is None:
                self.saver.save(sess, save_dir + self.save_filename)
        return min_loss

    def predict(self, dataset, save_dir):
        self.scaler = pickle_load(save_dir + self.save_scaler_filename)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            self.saver.restore(sess, save_dir + self.save_filename)

            xs = dataset2arrays(dataset, return_labels=False)
            self.scaler.transform(np.reshape(xs, (-1, 1)))
            batches_xs = batches(xs, self.conf["batch_size"])
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

        batches_train_xs = batches(xs, self.conf["batch_size"])
        batches_train_labels = batches(labels, self.conf["batch_size"])
        for train_x, train_label in zip(batches_train_xs, batches_train_labels):
            feed_dict = {
                self.tf_x: train_x,
                self.tf_label: train_label,
                self.is_training: is_training
            }
            self.global_step += 1
            if is_training:
                _, loss_val, summary = sess.run([self.optimizer, self.loss, self.merged_summary_op], feed_dict=feed_dict)
                self.summary_writer_train.add_summary(summary, self.global_step)
            else:
                loss_val, summary = sess.run([self.loss, self.merged_summary_op], feed_dict=feed_dict)
                self.summary_writer_dev.add_summary(summary, self.global_step)

            loss_batches.append(loss_val)
            size_batches.append(train_x.shape[0])

        loss = np.average(loss_batches, weights=size_batches)

        return loss


def smooth_p(p, eps=1e-3):
    return tf.maximum(eps, tf.minimum(1 - eps, p))
