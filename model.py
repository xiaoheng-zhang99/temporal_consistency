import tensorflow as tf
import numpy as np
from hyparams import hparams as hp
import os
import sys
import logging
import pandas as pd
from bert import modeling
# set logging
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.INFO,
                    datefmt='%I:%M:%S')

def bert_model(bert_config, is_training, input_ids, input_mask, segment_ids, use_one_hot_embeddings, use_sentence):
    """Creates a classification model."""
    model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)
    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    if use_sentence:
        output_layer = model.get_pooled_output()
    else:
        output_layer = model.get_sequence_output()
    return output_layer


class model:
    def __init__(self,
                 feats,
                 is_training=True):

        self.is_training = is_training
        self.feats = feats
        self.speech_input_dim=500
        self.text_input_dim = 150
        self._get_session()  # get session
        self._build_model()
        self._get_emo_iter()
        self.SEQ_DIM=768


    def _build_model(self):

        with tf.variable_scope('intput'):
            # current utt: Uc
            self.speech_input= tf.placeholder(dtype=tf.float32, shape=[None, None, self.speech_input_dim])
            # previous utt of target speaker: Up
            self.text_input = tf.placeholder(dtype=tf.float32, shape=[None, None, self.text_output_dim])
            self.groundtruths = tf.placeholder(dtype=tf.int64, shape=[None])

        with tf.variable_scope('SpeechEncoder', reuse=tf.AUTO_REUSE):
            ''' Previous utt of target speaker encoder '''
            cell = tf.nn.rnn.LSTMCell(self.SEQ_DIM)
            outputs, _ = tf.nn.dynamic_rnn(cell,
                                           inputs=self.speech_input,
                                           dtype=tf.float32)
            self.s_i=outputs

        with tf.variable_scope('TextEncoder', reuse=tf.AUTO_REUSE):
            ''' Opposite speaker encoder '''
            W1 = tf.Variable(tf.truncated_normal([1, 3, 5], stddev=0.1))
            conv1D = tf.keras.layers.Conv1D(1, W1, padding='valid')
            max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid')
            y = conv1D(self.text_input)
            y = max_pool_1d(y)
            self.t_j = outputs
        #dimension---add mask
        with tf.variable_scope('temporal alignement', reuse=tf.AUTO_REUSE):
            alpha=tf.tensordot(self.s_i,self.t_j, axes=1)
            alpha_prime = tf.nn.softmax(alpha)
            self.u_j = tf.reduce_sum(self.s_i * tf.expand_dims(alpha_prime, -1), 1)


        with tf.variable_scope('semantic_attention', reuse=tf.AUTO_REUSE):
            avg_tensor = [1, 2, 2, 1]
            self.h_t = tf.nn.avg_pool(value=self.t_j, ksize=avg_tensor, strides=avg_strides, padding="SAME")
            self.h_s = tf.nn.avg_pool(value=self.u_i, ksize=avg_tensor, strides=avg_strides, padding="SAME")

            hidden_size = self.out.shape[-1].value  # hidden size of the RNN layer
            attention_size = hp.ATTEN_SIZE
            # Trainable parameters
            x_s = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
            x_t = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
            b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
            V= tf.Variable(tf.random_normal([attention_size], stddev=0.1))

            wi_t = tf.tensordot(tf.nn.tanh(tf.tensordot(self.u_j, x_s, axes=1) +tf.tensordot(self.h_t, x_t, axes=1)  + b), V, axes=1)
            wi_s=tf.tensordot(tf.nn.tanh(tf.tensordot(self.t_j, x_t, axes=1) +tf.tensordot(self.h_s, x_s, axes=1)  + b), V, axes=1)
            wi_t_ = tf.nn.softmax(wi_t)
            wi_s_= tf.nn.softmax(wi_s)

            self.u_j_ = tf.reduce_sum(self.u_j * tf.expand_dims(wi_s_, -1), 1)
            self.t_i_ = tf.reduce_sum(self.t_i * tf.expand_dims(wi_t_, -1), 1)

        with tf.variable_scope('fusion_emo', reuse=tf.AUTO_REUSE):
            self.out = tf.concat([self.u_j_, self.t_i_], 1)
            cell = tf.nn.rnn.LSTMCell(self.SEQ_DIM)
            outputs, _ = tf.nn.dynamic_rnn(cell,
                                           inputs=self.out,
                                           dtype=tf.float32)
            self.F = outputs
            #FC
            out_weight1 = tf.get_variable('out_weight1', shape=[hp.SEQ_DIM * 3, hp.HIDDEN_DIM], dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            out_bias1 = tf.get_variable('out_bias1', shape=[hp.HIDDEN_DIM], dtype=tf.float32,
                                        initializer=tf.constant_initializer(0.1))
            dense=tf.matmul(self.out,out_weight1)+out_bias1
            dense = tf.nn.relu(dense)
            self.logits_emo = dense
        with tf.variable_scope('loss', reuse=tf.AUTO_REUSE):
            self.gt_emo = tf.one_hot(self.groundtruths, depth=4)
            self.gt_emo = label_smoothing(self.gt_emo)
            # classification loss
            self.emo_loss = tf.losses.softmax_cross_entropy(
                onehot_labels=self.gt_emo, logits=self.logits_emo)
            # total loss
            self.e_loss = self.emo_loss + hp.weight_decay * (tf.nn.l2_loss(out_weight1) + \
                                                             tf.nn.l2_loss(out_bias1) )
            self.e_optimizer = tf.train.AdamOptimizer(hp.lr).minimize(self.e_loss)

        #with tf.variable_scope('accuracy', reuse=tf.AUTO_REUSE):

        # Initialzation
        self.saver = tf.train.Saver(max_to_keep=2000)
        self.sess.run(tf.global_variables_initializer())

    def training(self):
        total_loss = 0
        total_acc = 0
        total_uar = 0
        Epoch = 1
        uar_list = []
        # start training
        for index in range(hp.num_train_steps):
            if index == 0:
                logging.info('=========training emotion classification !=========')

            try:
                current_utt, target_utt, opposite_utt, groundtruths = next(self.e_train_gen)
            except StopIteration:
                # generator has nothing left to generate
                # initialize iterator again
                logging.info('=========Epoch {} finished !========='.format(Epoch))
                Epoch += 1
                self._get_emo_iter()
                current_utt, target_utt, opposite_utt, groundtruths = next(self.e_train_gen)

            fd = {
                self.current_utt: current_utt,
                self.target_utt: target_utt,
                self.opposite_utt: opposite_utt,
                self.groundtruths: groundtruths
            }

            # uar
            pred_batch = self.sess.run(self.e_prediction, feed_dict=fd)
            uar_batch = recall_score(groundtruths, pred_batch, average='macro')
            # loss & acc
            loss_batch, _, acc_batch = self.sess.run([self.e_loss, self.e_optimizer, self.e_accuracy], feed_dict=fd)
            total_loss += loss_batch
            total_acc += acc_batch
            total_uar += uar_batch

            if (index + 1) % 20 == 0:
                logging.info(
                    'step: {}, Ave emo loss : {:.3f}, Ave emo train acc: {:.3f}, Ave emo train uar: {:.3f}'.format(
                        index + 1, total_loss / 20, total_acc / 20, total_uar / 20, ))
                total_loss = 0.0
                total_acc = 0.0
                total_uar = 0.0

            if (index + 1) % 100 == 0:
                self.save(index)
                test_gt, test_pred, ave_uar, ave_acc = self.testing()
                uar_list.append(float(ave_uar))

        logging.info('optimal step: %d, optimal uar: %.3f' % ((np.argmax(uar_list) + 1) * 100, max(uar_list)))

        return (np.argmax(uar_list) + 1) * 100

    def testing(self):
        self.is_training = False
        keep_proba = hp.keep_proba
        hp.keep_proba = 1
        # test data length

        df = pd.read_csv(hp.emo_test_file)
        self._get_emo_iter()
        test_gen = self.e_test_gen

        num_test_steps = len(df) // (2 * hp.BATCH_SIZE) + 1
        test_pred = []
        test_gt = []

        for i in range(num_test_steps):
            current_utt, target_utt, opposite_utt, groundtruths = next(test_gen)

            fd = {
                self.current_utt: current_utt,
                self.target_utt: target_utt,
                self.opposite_utt: opposite_utt,
                self.groundtruths: groundtruths
            }
            acc_batch, pred_batch = self.sess.run([self.e_accuracy, self.e_prediction], feed_dict=fd)
            uar_batch = recall_score(groundtruths, pred_batch, average='macro')
            test_pred += list(pred_batch)
            test_gt += list(groundtruths)

        ave_uar, ave_acc = evaluation(test_gt, test_pred)
        self.is_training = True
        hp.keep_proba = keep_proba

        return test_gt, test_pred, ave_uar, ave_acc

    def _get_session(self):
        self.sess = tf.Session()

    def save(self, e):
        if not os.path.exists(hp.model_path_save):
            os.makedirs(hp.model_path_save)
        self.saver.save(self.sess, hp.model_path_save + '/model_%d.ckpt' % (e + 1))

    def restore(self, e):
        self.saver.restore(self.sess, hp.model_path_load + '/model_%d.ckpt' % (e))