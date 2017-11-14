#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import re
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import yaml
import math
from input_helper import InputHelper
import gzip
import CWS

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_boolean("enable_word_embeddings", True, "Enable/disable the word embedding (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("channel_number", 2, "Number of word embeddings (default: 1)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("shared_units", 100, "neural units number in shared layer (default: 20)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("decay_coefficient", 5.0, "Decay coefficient (default: 2.5)")

#===================================================

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
    embedding_name = cfg['word_embeddings']['default']
    embedding_dimension = cfg['word_embeddings'][embedding_name[0]]['dimension']
else:
    embedding_dimension = FLAGS.embedding_dim

# Load data
print("Loading data...")
datasets = None
dataset_names = cfg["datasets"]["default"]
multi_task_num = len(dataset_names)
input_helper = InputHelper()
train, dev, vocab_processor, sum_no_batches = input_helper.get_datasets(percent_dev = 0.1,
                                            batch_size = FLAGS.batch_size, dataset_names = dataset_names )

# Training
# ==================================================
print("Start building")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=len(train[0][0][0]),
            num_classes=2,#????????????
            vocab_size=len(vocab_processor.vocabulary_),
            multi_size = multi_task_num,
            embedding_size=embedding_dimension,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            shared_units=FLAGS.shared_units,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            channel = FLAGS.channel_number)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(cnn.learning_rate)
        print("initialized cnn objects")

    train_op=[]
    for i  in range(multi_task_num):
        grads_and_vars = optimizer.compute_gradients(cnn.loss[i])
        train_op.append(optimizer.apply_gradients(grads_and_vars, global_step=global_step))
    print ("defined training operation")

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join("/home/xxy/Desktop/mvcnn/runs/multi-task", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary=[]
    acc_summary=[]
    for i in range(multi_task_num):
        loss_summary.append(tf.summary.scalar("loss", cnn.loss[i]))
        acc_summary.append(tf.summary.scalar("accuracy", cnn.accuracy[i]))

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Write vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    if FLAGS.enable_word_embeddings and cfg['word_embeddings']['default'] is not None:
        vocabulary = vocab_processor.vocabulary_
        initW = None
        w_list=[]
        for e in embedding_name:
            if e == 'word2vec':
                # load embedding vectors from the word2vec
                print("Load word2vec file {}".format(cfg['word_embeddings']['word2vec']['path']))
                w = data_helpers.load_embedding_vectors_word2vec(vocabulary,
                                                                     cfg['word_embeddings']['word2vec']['path'],
                                                                     cfg['word_embeddings']['word2vec']['binary'])
                print("word2vec file has been loaded")
            elif e == 'glove':
                # load embedding vectors from the glove
                print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
                w = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
                print("glove file has been loaded\n")

            w_list.append(w)

        if FLAGS.channel_number > 1:
            initW = w_list[0]
            for i in range(FLAGS.channel_number-1):
                initW = np.dstack((initW, w_list[i+1]))
        else:
            initW = w
        sess.run(cnn.W.assign(initW))

    def train_step(x_batch, y_batch, learning_rate, type_index):
        """
        A single training step
        """
        feed_dict = {
          cnn.input_x: x_batch,
          #cnn.input_y: y_batch,
          cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
          cnn.learning_rate: learning_rate,
        }
        for i in range(multi_task_num):
            if i == type_index:
                feed_dict[cnn.input_y[i]] = y_batch
            else: feed_dict[cnn.input_y[i]] = np.zeros((len(x_batch),2))

        _, step, summaries, loss, accuracy, pred = sess.run(
            [train_op[type_index], global_step, train_summary_op, cnn.loss[type_index],
             cnn.accuracy[type_index],cnn.predictions[type_index]], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: type {}, step {}, loss {:g}, acc {:g}, learning_rate {:g}"
              .format(time_str, type_index, step, loss, accuracy, learning_rate))
        #train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch, type_index, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          cnn.input_x: x_batch,
          #cnn.input_y: y_batch,
          cnn.dropout_keep_prob: 1.0,
        }
        for i in range(multi_task_num):
            if i == type_index:
                feed_dict[cnn.input_y[i]] = y_batch
            else: feed_dict[cnn.input_y[i]] = np.zeros((len(x_batch),2))
        step, summaries, loss, accuracy, pred = sess.run(
            [global_step, dev_summary_op, cnn.loss[type_index], cnn.accuracy[type_index],
            cnn.predictions[type_index]], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: type{}, step {}, loss {:g}, acc {:g}".format(time_str, type_index, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)
        return accuracy

    # Generate batches
    batches = []
    for i in range(multi_task_num):
        batches.append(data_helpers.batch_iter(
                list(zip(train[i][0], train[i][1])), FLAGS.batch_size, FLAGS.num_epochs))
    # It uses dynamic learning rate with a high value at the beginning to speed up the training
    max_learning_rate = 0.005
    min_learning_rate = 0.0001
    decay_speed = FLAGS.decay_coefficient*len(train[0][0])/FLAGS.batch_size

    # Training loop. For each batch...
    counter = 0
    for n in range(sum_no_batches * FLAGS.num_epochs):
        #idx = round(np.random.uniform(low=0, high=multi_task_num))
        #if idx < 0 or idx > multi_task_num - 1: continue
        #type_index = int(idx)
        type_index = counter%multi_task_num
        batch = batches[type_index].__next__()
        if len(batch)<1: continue

        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
        counter += 1
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch, learning_rate, type_index)
        current_step = tf.train.global_step(sess, global_step)

        acc=[]
        if current_step % FLAGS.evaluate_every == 0:
            for i in range(multi_task_num):
                print("\nEvaluation:")
                '''
                dev_batches = data_helpers.batch_iter(list(zip(dev[i][0], dev[i][1])),
                                                      2 * FLAGS.batch_size, 1)
                for db in dev_batches:
                    x_dev, y_dev = zip(*db)
                    dev_step(x_dev, y_dev, type_index=i, writer=dev_summary_writer)
                '''
                x_dev, y_dev = dev[i][0], dev[i][1]
                acc.append(dev_step(x_dev, y_dev, type_index=i, writer=dev_summary_writer))
                print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph" + str(n) + ".pb", as_text=False)
            print("Saved model checkpoint to {}\n".format(path))
