# (C) 2018-present Klebert Engineering

# ===============================[ Imports ]=============================

import numpy as np
import tensorflow as tf

# ============================[ Local Imports ]==========================

from deepspell.models import encoder
from deepspell_optimization.models import optimizer


# =======================[ LSTM Extrapolator Model ]=====================

class DSVariationalLstmAutoEncoderOptimizer(optimizer.DSModelOptimizerMixin, encoder.DSVariationalLstmAutoEncoder):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class"""
        encoder.DSVariationalLstmAutoEncoder.__init__(
            self,
            name_scope="spelling-encoder",
            version=1,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            args_to_update=kwargs)
        optimizer.DSModelOptimizerMixin.__init__(self, kwargs)

        # -- Read params
        self.decoder_state_size_per_layer = kwargs.pop("decoder_state_size_per_layer", [128, 128])
        self.decoder_input_keep_prob = kwargs.pop("decoder_input_keep_prob", .75)
        self.kl_rate_rise_iterations = kwargs.pop("kl_rate_rise_iterations", 2000)
        self.kl_rate_rise_threshold = kwargs.pop("kl_rate_rise_threshold", 1000)
        self.current_kl_rate = kwargs.pop("current_kl_rate", .0)
        self.latent_space_as_decoder_state = kwargs.pop("latent_space_as_decoder_state", True)

        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
            (self.tf_eol_char_id,
             self.tf_unk_char_id,
             self.tf_start_char_id,
             self.tf_correct_decoder_output,
             self.tf_stepwise_decoder_output) = self._latent_to_stepwise_decoder()
            (self.tf_train_op,
             self.tf_lexical_loss_summary) = self._encoder_decoder_optimizer()

        self._finish_init_optimizer()

    def train(self, training_corpus, sample_grammar, train_test_split=None):
        batch, lengths, corrupted_batch, corrupted_lengths = None, None, None, None

        def variational_rnn_feed_fn(epoch_it, learning_rate):
            global batch, lengths, corrupted_batch, corrupted_lengths
            batch, lengths, epoch_it, corrupted_batch, corrupted_lengths = training_corpus.next_batches_and_lengths(
                self.batch_size,
                sample_grammar,
                epoch_it,
                corrupt=True,
                embed_with_class=False)
            return {
                self.tf_corrupt_encoder_input: corrupted_batch,
                self.tf_timesteps_per_batch: corrupted_lengths,
                self.tf_learning_rate: learning_rate,
                self.tf_kl_rate: self._next_kl_rate(),
                self.tf_eol_char_id: self.featureset.charset_eol_index,
                self.tf_unk_char_id: self.featureset.charset_unk_index,
                self.tf_start_char_id: self.featureset.charset_bol_index,
                self.tf_correct_decoder_output: batch
            }, epoch_it

        def run_callback_fn(results):
            global batch, lengths, corrupted_batch, corrupted_lengths
            if (self.iteration % 100) == 0:
                stepwise_decoder_output = results[1]
                print("\nSome decoded vocabulary at iteration {}:".format(self.iteration))
                for corrupt_sample, corrupt_len, decoded_sample, correct_sample, correct_len in zip(
                        corrupted_batch[:20], corrupted_lengths, stepwise_decoder_output, batch, lengths):
                    print("   {} -> {} ({})".format(
                        "".join(self.featureset.charset[np.argmax(embedding)] for embedding in corrupt_sample[:int(corrupt_len)]),
                        "".join(self.featureset.charset[np.argmax(embedding)] for embedding in decoded_sample[:int(correct_len)]),
                        "".join(self.featureset.charset[np.argmax(embedding)] for embedding in correct_sample[:int(correct_len)])
                    ))
                print("")

        self._train(
            [self.tf_train_op, self.tf_stepwise_decoder_output],
            [self.tf_kl_loss_summary, self.tf_lexical_loss_summary],
            training_corpus, sample_grammar, train_test_split,
            feed_fn=variational_rnn_feed_fn,
            run_callback_fn=run_callback_fn)

    def generate_name(self):
        return super().generate_name() + "_emb{}_fw{}_bw{}_co{}_de{}{}_drop{}".format(
            self.embedding_size,
            "-".join(str(n) for n in self.encoder_fw_state_size_per_layer),
            "-".join(str(n) for n in self.encoder_bw_state_size_per_layer),
            "-".join(str(n) for n in self.encoder_combine_state_size_per_layer),
            "st" if self.latent_space_as_decoder_state else "in",
            "-".join(str(n) for n in self.decoder_state_size_per_layer),
            int((1.0-self.decoder_input_keep_prob)*100)
        )

    def info(self):
        result = super().info()
        result["encoder_fw_state_size_per_layer"] = self.encoder_fw_state_size_per_layer
        result["encoder_bw_state_size_per_layer"] = self.encoder_bw_state_size_per_layer
        result["encoder_combine_state_size_per_layer"] = self.encoder_combine_state_size_per_layer
        result["decoder_state_size_per_layer"] = self.decoder_state_size_per_layer
        result["embedding_size"] = self.embedding_size
        result["latent_space_as_decoder_state"] = self.latent_space_as_decoder_state
        result["decoder_input_keep_prob"] = self.decoder_input_keep_prob
        result["current_kl_rate"] = self.current_kl_rate
        result["kl_rate_rise_iterations"] = self.kl_rate_rise_iterations
        result["kl_rate_rise_threshold"] = self.kl_rate_rise_threshold
        return result

    # ----------------------[ Private Methods ]----------------------

    def _next_kl_rate(self):
        if self.iteration < self.kl_rate_rise_threshold:
            self.current_kl_rate = .0
        else:
            self.current_kl_rate = min(
                1.0,
                (self.iteration-self.kl_rate_rise_threshold)/float(self.kl_rate_rise_iterations))
        return self.current_kl_rate

    def _latent_to_stepwise_decoder(self):
        """
        :return: tf_eol_char_id, tf_unk_char_id, tf_start_char_id, tf_correct_decoder_output, tf_stepwise_decoder_output
        """
        with tf.variable_scope('stepwise_decoder'):
            tf_eol_char_id = tf.placeholder(tf.int32, shape=[])
            tf_unk_char_id = tf.placeholder(tf.int32, shape=[])
            tf_start_char_id = tf.placeholder(tf.int32, shape=[])
            tf_correct_decoder_output = tf.placeholder(
                dtype=tf.float32,
                shape=(None, None, self.featureset.num_lexical_features()))
            tf_batch_size = tf.shape(tf_correct_decoder_output)[0]
            tf_decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                    self.decoder_state_size_per_layer]),
                self.featureset.num_lexical_features())

            if self.latent_space_as_decoder_state:
                with tf.variable_scope('latent_to_decoder'):
                    concat_state_size = sum(n*2 for n in self.decoder_state_size_per_layer)
                    tf_w = tf.get_variable("w", [self.embedding_size, concat_state_size], dtype=tf.float32)
                    tf_b = tf.get_variable("b", [concat_state_size], dtype=tf.float32)
                    tf_decoder_initial_state = self._prelu(tf.matmul(self.tf_latent_random_vectors, tf_w) + tf_b)
                    tf_decoder_initial_state_tuple_list, pos_in_state = [], 0
                    for state_size in self.decoder_state_size_per_layer:
                        tf_decoder_initial_state_tuple_list += [tf.contrib.rnn.LSTMStateTuple(
                            tf_decoder_initial_state[:, pos_in_state:pos_in_state+state_size],
                            tf_decoder_initial_state[:, pos_in_state+state_size:pos_in_state+2*state_size])]
                        pos_in_state += state_size*2  # *2 for state+mem
            else:
                tf_decoder_initial_state_tuple_list = tf_decoder_cell.zero_state(tf_batch_size, dtype=tf.float32)

            tf_max_decoder_steps = tf.shape(tf_correct_decoder_output)[1]
            tf_t = tf.constant(0)
            tf_stepwise_decoder_output = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            tf_prev_output = tf.reshape(tf.tile(
                tf.one_hot(tf_start_char_id, depth=self.featureset.num_lexical_features()),
                [tf_batch_size]), shape=(-1, self.featureset.num_lexical_features()))

            def should_continue(t, *_):
                return t < tf_max_decoder_steps

            def iteration(t, prev_output, state, stepwise_decoder_output):
                input_keep_prob = tf.random_uniform([], .0, 1.)
                prev_output = tf.cond(
                    tf.logical_and(input_keep_prob > self.decoder_input_keep_prob, t > 0),
                    lambda: tf.reshape(tf.tile(
                        tf.one_hot(tf_unk_char_id, depth=self.featureset.num_lexical_features()),
                        [tf_batch_size]), shape=(-1, self.featureset.num_lexical_features())),
                    lambda: prev_output)
                prev_output = tf.concat([prev_output, self.tf_latent_random_vectors], axis=1)
                prev_output, state = tf_decoder_cell(state=state, inputs=prev_output)
                stepwise_decoder_output = stepwise_decoder_output.write(t, prev_output)
                prev_output = tf_correct_decoder_output[:, t, :]
                return t+1, prev_output, state, stepwise_decoder_output

            _, _, _, tf_stepwise_decoder_output = tf.while_loop(
                should_continue,
                iteration,
                [tf_t, tf_prev_output, tuple(tf_decoder_initial_state_tuple_list), tf_stepwise_decoder_output])

        tf_stepwise_decoder_output = tf.transpose(tf_stepwise_decoder_output.stack(), perm=(1, 0, 2))
        return tf_eol_char_id, tf_unk_char_id, tf_start_char_id, tf_correct_decoder_output, tf_stepwise_decoder_output

    def _encoder_decoder_optimizer(self):
        """
        :return: tf_train_op, tf_kd_loss_summary, tf_lexical_loss_summary, tf_unk_lexical_idx, tf_stepwise_decoder_output
        """
        with tf.name_scope("decoder_optimizer"):
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
            tf_optimizer = tf.train.RMSPropOptimizer(self.tf_learning_rate)
            tf_train_op = tf.contrib.layers.optimize_loss(
                loss=tf_lexical_loss + self.tf_kl_loss,
                global_step=global_step,
                learning_rate=None,
                summaries=[],
                optimizer=tf_optimizer)

        return tf_train_op, tf_lexical_loss_summary
