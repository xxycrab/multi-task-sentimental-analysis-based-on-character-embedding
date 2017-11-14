import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, multi_size, vocab_size,
      embedding_size, filter_sizes, num_filters, shared_units, Cemb, l2_reg_lambda=0.0, channel=1):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y=[]
        for i in range(multi_size):
            self.input_y.append(tf.placeholder(tf.float32, [None, num_classes], name="input_y"+str(i)))
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = []
        for i in range(multi_size):
            l2_loss.append(tf.constant(0.0, name = "l2_loss"+str(i)))

        # Embedding layer
        self.channel = channel
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size, self.channel], -1.0, 1.0),
                name="W")
            self.embedded_list = []
            for i in range(self.channel):
                self.embedded_chars = tf.nn.embedding_lookup(self.W[:,:,i], self.input_x)
                print(self.W[:, :, i].shape)
                self.embedded_list.append(self.embedded_chars)
            if self.channel>1:
                self.embedded_chars_expanded = tf.stack(self.embedded_list, axis = -1)
            elif self.channel == 1:
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 2, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        self.scores=[]
        self.predictions=[]
        self.loss=[]
        self.accuracy=[]

        # softmax and output layer
        for i in range(multi_size):
            with tf.name_scope("output"+str(i)):
                W = tf.get_variable(
                    "W"+str(i),
                    shape=[num_filters_total, shared_units],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[shared_units]), name="b"+str(i))
                l2_loss[i] += tf.nn.l2_loss(W)
                l2_loss[i] += tf.nn.l2_loss(b)
                inference = tf.nn.softmax(tf.nn.xw_plus_b(self.h_drop, W, b, name="softmax"+str(i)))
                inference = tf.nn.dropout(inference, self.dropout_keep_prob)

                W2 = tf.get_variable(
                    "W2" + str(i),
                    shape=[shared_units, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2" + str(i))
                l2_loss[i] += tf.nn.l2_loss(W2)
                l2_loss[i] += tf.nn.l2_loss(b2)

                self.scores.append(tf.nn.xw_plus_b(inference, W2, b2, name="scores"+str(i)))
                self.predictions.append(tf.argmax(tf.nn.softmax(self.scores[i]), 1, name="predictions"+str(i)))

        # CalculateMean cross-entropy loss
        for i in range(multi_size):
            with tf.name_scope("loss"+str(i)):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores[i], labels=self.input_y[i])
                self.loss.append(tf.reduce_mean(losses) + l2_reg_lambda * l2_loss[i])

        # Accuracy
        for i in range(multi_size):
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions[i], tf.argmax(self.input_y[i], 1))
                self.accuracy.append(tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy"+str(i)))
