# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import tensorflow as tf
import numpy as np

# ============================[ Local Imports ]==========================

from . import predictor
from . import featureset

# =======================[ LSTM Extrapolator Model ]=====================


class DSVariationalLstmAutoEncoder(predictor.DSPredictor):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class"""
        super().__init__(
            name_scope="spelling-encoder",
            version=1,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            kwargs_to_update=kwargs)

        # -- Read params
        self.encoder_fw_state_size_per_layer = kwargs.pop("encoder_fw_state_size_per_layer", [128, 128])
        self.encoder_bw_state_size_per_layer = kwargs.pop("encoder_bw_state_size_per_layer", [128, 128])
        self.decoder_state_size_per_layer = kwargs.pop("decoder_state_size_per_layer", [128, 128])
        self.embedding_size = kwargs.pop("embedding_size", 8)
        self.decoder_input_keep_prob = kwargs.pop("decoder_input_keep_prob", .75)

        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
            self.tf_encoder_final_state = self._encoder()
            (self.tf_latent_vector,
             self.tf_latent_mean,
             self.tf_kl_loss,
             self.tf_kl_loss_summary,
             self.tf_kl_rate) = self._encoder_to_latent()
            (self.tf_eol_char_id,
             self.tf_unk_char_id,
             self.tf_start_char_id,
             self.tf_correct_decoder_output,
             self.tf_stepwise_decoder_output) = self._latent_to_stepwise_decoder()
            (self.tf_train_op,
             self.tf_lexical_loss_summary) = self._encoder_decoder_optimizer()
        self._finish_init()

    def train(self, training_corpus, sample_grammar, train_test_split=None):
        self._train(
            self.tf_train_op,
            [self.tf_kl_loss_summary, self.tf_lexical_loss_summary],
            training_corpus, sample_grammar, train_test_split)

    def name(self):
        return super().name()+"_emb{}_fw{}_bw{}_de{}_drop{}".format(
            self.embedding_size,
            "-".join(str(n) for n in self.encoder_fw_state_size_per_layer),
            "-".join(str(n) for n in self.encoder_bw_state_size_per_layer),
            "-".join(str(n) for n in self.decoder_state_size_per_layer),
            int(self.decoder_input_keep_prob*100)
        )

    def info(self):
        result = super().info()
        result["encoder_fw_state_size_per_layer"] = self.encoder_fw_state_size_per_layer
        result["encoder_bw_state_size_per_layer"] = self.encoder_bw_state_size_per_layer
        result["decoder_state_size_per_layer"] = self.decoder_state_size_per_layer
        result["embedding_size"] = self.embedding_size
        result["decoder_input_keep_prob"] = self.decoder_input_keep_prob
        return result

    def encode(self):
        pass

    # ----------------------[ Private Methods ]----------------------

    @staticmethod
    def _prelu(x):
        with tf.variable_scope("prelu"):
            alphas = tf.get_variable(
                'alpha', x.get_shape()[-1],
                initializer=tf.constant_initializer(0.0),
                dtype=tf.float32)
            pos = tf.nn.relu(x)
            neg = alphas * (x - abs(x)) * 0.5
            return pos + neg

    def _encoder(self):
        """
        :return: tf_encoder_final_fw_state_tuple_stack, tf_encoder_final_bw_state_tuple_stack
        """
        # -- Input placeholders: batch of training sequences and their lengths
        with tf.name_scope("encoder"):

            # -- Slice lexical features from lexical-logical input
            tf_lexical_embeddings_per_timestep_per_batch = self.tf_lexical_logical_embeddings_per_timestep_per_batch[
                                                           :, :, :self.num_lexical_features]

            # -- Backward pass
            tf_encoder_backward_cell = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                self.encoder_bw_state_size_per_layer])
            tf_backward_embeddings_per_timestep_per_batch, tf_final_bw_state_tuple_stack = tf.nn.dynamic_rnn(
                cell=tf_encoder_backward_cell,
                inputs=tf.reverse(tf_lexical_embeddings_per_timestep_per_batch, axis=[1]),
                initial_state=tf_encoder_backward_cell.zero_state(
                    self.tf_lexical_logical_embeddings_per_timestep_per_batch_shape[0],
                    tf.float32),
                time_major=False)

            # -- Forward pass
            tf_discriminator_forward_cell = tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                    self.encoder_fw_state_size_per_layer])
            # -- Create a dynamically unrolled RNN to produce the character category discrimination
            tf_logical_predictions_per_timestep_per_batch, tf_final_fw_state_tuple_stack = tf.nn.dynamic_rnn(
                cell=tf_discriminator_forward_cell,
                inputs=tf.concat([
                    tf_lexical_embeddings_per_timestep_per_batch,
                    tf.reverse(tf_backward_embeddings_per_timestep_per_batch, axis=[1])], axis=2),
                sequence_length=self.tf_timesteps_per_batch,
                initial_state=tf_discriminator_forward_cell.zero_state(
                    self.tf_lexical_logical_embeddings_per_timestep_per_batch_shape[0],
                    tf.float32),
                time_major=False)

        tf_final_encoder_state = tf.concat([
            state
            for state_tuple_stack in (tf_final_fw_state_tuple_stack, tf_final_bw_state_tuple_stack)
            for state_tuple in state_tuple_stack
            for state in state_tuple], axis=0)

        return tf_final_encoder_state

    def _encoder_to_latent(self):
        """
        :return: tf_latent_mean, tf_latent_variance, tf_latent_vec
        """
        with tf.variable_scope('encoder_to_latent'):
            kl_rate = tf.placeholder(tf.float32)
            concat_state_size = sum(
                n*2 for n in self.encoder_fw_state_size_per_layer+self.encoder_bw_state_size_per_layer)
            w = tf.get_variable("w", [concat_state_size, 2 * self.embedding_size], dtype=tf.float32)
            b = tf.get_variable("b", [2 * self.embedding_size], dtype=tf.float32)
            mean_logvar = self._prelu(tf.matmul(self.tf_encoder_final_state, w) + b)
            mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
            noise = tf.random_normal(tf.shape(mean))
            sampled_vector = mean + tf.exp(0.5 * logvar) * noise
            kl_loss = tf.reduce_mean(-0.5 * (logvar - tf.square(mean) - tf.exp(logvar) + 1.0))
            # if kl_min:
            #     kl_loss = tf.reduce_sum(tf.maximum(kl_ave, kl_min))
            kl_loss *= kl_rate
            kl_loss_summary = tf.summary.scalar("kl_loss", kl_loss)
            return sampled_vector, mean, kl_loss, kl_loss_summary, kl_rate

    def _latent_to_stepwise_decoder(self):
        """
        :return: tf_eol_char_id, tf_unk_char_id, tf_start_char_id, tf_correct_decoder_output, tf_stepwise_decoder_output
        """
        with tf.variable_scope('latent_to_decoder'):
            concat_state_size = sum(n*2 for n in self.decoder_state_size_per_layer)
            tf_w = tf.get_variable("w", [self.embedding_size, concat_state_size], dtype=tf.float32)
            tf_b = tf.get_variable("b", [concat_state_size], dtype=tf.float32)
            tf_decoder_initial_state = self._prelu(tf.matmul(self.tf_latent_vector, tf_w) + tf_b)
            tf_decoder_initial_state_tuple_list, pos_in_state = [], 0
            for state_size in self.decoder_state_size_per_layer:
                tf_decoder_initial_state_tuple_list += [tf.contrib.rnn.LSTMStateTuple(
                    tf_decoder_initial_state[pos_in_state:pos_in_state+state_size],
                    tf_decoder_initial_state[pos_in_state+state_size:pos_in_state+2*state_size])]
                pos_in_state += state_size*2  # *2 for state+mem

        with tf.variable_scope('stepwise_decoder'):
            tf_eol_char_id = tf.placeholder(tf.int32)
            tf_unk_char_id = tf.placeholder(tf.int32)
            tf_start_char_id = tf.placeholder(tf.int32)
            tf_correct_decoder_output = tf.placeholder(
                dtype=tf.float32,
                shape=(-1, self.featureset.num_lexical_features()))
            tf_decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                    self.decoder_state_size_per_layer]),
                self.featureset.num_lexical_features())
            tf_max_decoder_steps = tf.shape(tf_correct_decoder_output)[0]
            tf_t = tf.constant(0)
            tf_stepwise_decoder_output = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            tf_prev_output = tf.one_hot(tf_start_char_id, depth=self.featureset.num_lexical_features())

            def should_continue(t, prev_output, *_):
                return tf.logical_and(
                    t < tf_max_decoder_steps,
                    prev_output != tf.one_hot(tf_eol_char_id, depth=self.featureset.num_lexical_features()))

            def iteration(t, prev_output, state, stepwise_decoder_output):
                input_keep_prob = tf.random_uniform([], .0, 1.)
                prev_output = tf.cond(
                    input_keep_prob > self.decoder_input_keep_prob,
                    lambda: tf.one_hot(tf_unk_char_id, depth=self.num_lexical_features()),
                    lambda: prev_output)
                prev_output, state = tf_decoder_cell(state=state, inputs=prev_output)
                stepwise_decoder_output = stepwise_decoder_output.write(t, prev_output)
                prev_output = tf_correct_decoder_output[t]
                return t+1, prev_output, state, stepwise_decoder_output

            _, _, _, tf_stepwise_decoder_output = tf.while_loop(
                should_continue,
                iteration,
                [tf_t, tf_prev_output, tf_decoder_initial_state_tuple_list, tf_stepwise_decoder_output])

        tf_stepwise_decoder_output = tf_stepwise_decoder_output.stack()
        return tf_eol_char_id, tf_unk_char_id, tf_start_char_id, tf_correct_decoder_output, tf_stepwise_decoder_output

    def _encoder_decoder_optimizer(self):
        """
        :return: tf_train_op, tf_kd_loss_summary, tf_lexical_loss_summary, tf_unk_lexical_idx, tf_stepwise_decoder_output
        """
        with tf.name_scope("extrapolator_optimizer"):
            # -- Obtain global training step
            global_step = tf.contrib.framework.get_global_step()

            # -- Calculate the average cross entropy for the lexical classes per timestep
            tf_lexical_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.tf_correct_decoder_output,
                logits=self.tf_stepwise_decoder_output,
                dim=2))

            # -- Create summaries for TensorBoard
            tf_lexical_loss_summary = tf.summary.scalar("lexical_loss", tf_lexical_loss)

            # -- Define training op
            optimizer = tf.train.RMSPropOptimizer(self.tf_learning_rate)
            tf_train_op = tf.contrib.layers.optimize_loss(
                loss=tf_lexical_loss + self.tf_kl_loss,
                global_step=global_step,
                learning_rate=None,
                summaries=[],
                optimizer=optimizer)

        return tf_train_op, tf_lexical_loss_summary
