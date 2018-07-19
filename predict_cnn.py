# -*- coding:utf-8 -*-

import os
import sys
import time
import tensorflow as tf
import data_helpers as dh
import json

# Parameters
# ==================================================

id_to_cat = json.load(open("./json/category_id.json", 'r', encoding='utf-8'))['id_to_cat']


logger = dh.logger_fn('tflog', 'logs/predict-{0}.log'.format(time.asctime()))

# Data Parameters
tf.flags.DEFINE_string("training_data_file", "./data/train_data_set.txt", "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", "./data/val_data_set.txt", "Data source for the validation data")
tf.flags.DEFINE_string("test_data_file", "./data/test_data_set.txt", "Data source for the test data")
tf.flags.DEFINE_string("predict_data_file", "./data/predict_data.txt", "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", "./", "Checkpoint directory from training run")
# tf.flags.DEFINE_string("vocab_data_file", "./", "Vocabulary file")

# Model Hyperparameters
# tf.flags.DEFINE_integer("pad_seq_len", 100, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 32, "Number of labels (depends on the task)")
tf.flags.DEFINE_integer("top_num", 5, "Number of top K prediction classes (default: 5)")
tf.flags.DEFINE_float("threshold", 0.5, "Threshold for prediction classes (default: 0.5)")

# Test Parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
para_key_values = FLAGS.__flags

logger = dh.logger_fn('tflog', 'logs/predict-{0}.log'.format(time.asctime()))
logger.info("input parameter:")
parameter_info = " ".join(["\nparameter: {0:<30} value: {1:<50}".format(key, val) for key, val in para_key_values.items()])
logger.info(parameter_info)


print("load train and val data sets.....")
logger.info('✔︎ Test data processing...')
x_train, y_train = dh.process_file(FLAGS.training_data_file)
x_val, y_val = dh.process_file(FLAGS.validation_data_file)
x_test, y_test = dh.process_file(FLAGS.test_data_file)

# 得到所有数据中最长文本长度
pad_seq_len = dh.get_pad_seq_len(x_train, x_val, x_test)

# 将数据pad为统一长度，同时对label进行0，1编码
x_predict = dh.process_data_for_predict(FLAGS.predict_data_file, pad_seq_len)


def predict():
    """Predict Use TextCNN model."""

    # Load cnn model
    logger.info("✔ Loading model...")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            feed_dict = {
                input_x: x_predict,
                dropout_keep_prob: 1.0,
            }
            batch_scores = sess.run(scores, feed_dict)
            predicted_labels_threshold, predicted_values_threshold = \
                dh.get_label_using_scores_by_threshold(scores=batch_scores, threshold=FLAGS.threshold)

            # print(predicted_labels_threshold, predicted_values_threshold)
            all_threshold = []
            for _ in predicted_labels_threshold:
                temp = []
                for id in _:
                    temp.append(id_to_cat[str(id)])
                all_threshold.append(temp)
            print(all_threshold)

            # Predict by topK
            all_topK = []
            predicted_labels_topk, predicted_values_topk = \
                dh.get_label_using_scores_by_topk(batch_scores, top_num=FLAGS.top_num + 1)
            for _ in predicted_labels_topk:
                temp = []
                for id in _:
                    temp.append(id_to_cat[str(id)])
                all_topK.append(temp)
            print(all_topK)
    logger.info("✔ Done.")


if __name__ == '__main__':
    predict()
