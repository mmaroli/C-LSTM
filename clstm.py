"""
C-LSTM Model as mentioned in:
A C-LSTM Neural Network for Text Classification
https://arxiv.org/pdf/1511.08630.pdf
"""

import os
import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from create_train_dev_split import get_train_data


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("train_data", None, "Text to classify")
flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries.")
flags.DEFINE_integer(
    "epochs_to_train", 50,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("max_steps", 10000, "Max steps before training stops")


class C_LSTM(object):
    def __init__(self, session, embedding_matrix):
        self._session = session
        self._embedding_matrix = embedding_matrix
        self._loss = None

    def get_loss(self):
        return self._loss


    def clstm_model(self, embedding_matrix=None):
        """ Create model for C-LSTM

        Args:
            embedding_matrix: Embedding matrix of documents being classified
                              with shape [batch, L, D, channels]

        Returns:
            Logits

        """
        if embedding_matrix is None:
            embedding_matrix = self._embedding_matrix

        with tf.variable_scope('conv') as scope:
            filter_height = 3
            num_filters = 64
            kernel = tf.get_variable(name='weights', shape=[filter_height,
                    embedding_matrix.shape[2], 1, num_filters],
                    initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32),
                    dtype=tf.float32)
            conv = tf.nn.conv2d(input=embedding_matrix, filter=kernel, strides=[1,1,1,1], padding="VALID")
            biases = tf.get_variable(name='biases', shape=[num_filters], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)


        # concatenate feature maps W = [c1; c2; c3] column wise
        # then each row of W represents feature representation for window vector
        # row Wj = feature representation for window vector at position j

        with tf.variable_scope('lstm') as scope:
            W = tf.reshape(conv1, [conv1.shape[1], conv1.shape[3]])
            inputs = []
            for j in range(conv1.shape[1]):
                inputs.append( tf.reshape(W[j,:], [1, conv1.shape[3]]) )
            # each row of W is fed into LSTM as a timestep
            cell = tf.nn.rnn_cell.LSTMCell(num_units=64, name=scope.name)
            #inputs: list of inputs of tensors with [batch_size, input_size]
            # list of length conv1.shape[1] with tensors [1, 64]
            outputs, state = tf.nn.static_rnn(cell=cell, inputs=inputs, dtype=tf.float32, scope='lstm')

        with tf.variable_scope('softmax') as scope:
            num_units = 64
            num_classes = 5
            weights = tf.get_variable(name='weights', shape=[num_units, num_classes],
                                      initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            biases = tf.get_variable(name='baises', shape=[num_classes], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            softmax_linear = tf.nn.bias_add(tf.matmul(outputs[-1], weights), biases, name=scope.name)
            return softmax_linear


    def loss(self, logits, labels):
        """ Loss
            Args:
                logits: Logits from clstm()
                labels: 1-D tensor of shape [batch_size]
        """
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        self._loss = cross_entropy_mean
        return cross_entropy_mean


    def train(self, total_loss, global_step):
        """ Training process """
        lr = FLAGS.learning_rate
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

        apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)


        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
          0.99, global_step)
        with tf.control_dependencies([apply_gradients_op]):
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

        return variables_averages_op



def main(argv=None):
    with tf.Graph().as_default(), tf.Session() as session:

        embedding_matrix, labels = get_train_data()
        embedding_matrix = tf.expand_dims(embedding_matrix, -1) # channels

        train_dataset = tf.data.Dataset.from_tensor_slices( (embedding_matrix, tf.constant(labels)) ).repeat(FLAGS.epochs_to_train)
        iterator = train_dataset.make_one_shot_iterator()
        element, label = iterator.get_next()
        label = label - 1 # to make in range of [0,num_classes)
        next_element = tf.expand_dims(element, 0)
        next_label = tf.expand_dims(label, 0)


        global_step = tf.train.get_or_create_global_step()


        model = C_LSTM(session, next_element)
        logits = model.clstm_model()
        loss = model.loss(logits, next_label)
        train_op = model.train(loss, global_step)


        # For tensorboard
        writer = tf.summary.FileWriter('tensorboard')
        writer.add_graph(tf.get_default_graph())
        writer.flush()


        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss) # gets loss

            def after_run(self, run_context, run_values):
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value = run_values.results
                print(f"{datetime.now()} step: {self._step}, loss: {loss_value}")

        max_steps = FLAGS.max_steps
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir='checkpoints',
            hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook()]
        ) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


if __name__ == '__main__':
    tf.app.run()
