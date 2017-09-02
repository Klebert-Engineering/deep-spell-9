# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import tensorflow as tf

# ============================[ Local Imports ]==========================

from . import abstract


# ======================[ LSTM Discriminator Model ]=====================

class DSLstmDiscriminator(abstract.DSPredictor):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class
        :param total_features_per_character: The exact number of features per character,
         which is required for the Graph construction.
        """
        super().__init__("discriminator", 1, file_or_folder, log_dir, kwargs)
        # -- Read params
        self.fw_state_size_per_layer = kwargs.pop("fw_state_size_per_layer", [128, 128])
        self.bw_state_size_per_layer = kwargs.pop("bw_state_size_per_layer", [128, 128])
        self.min_sample_length_before_truncation = kwargs.pop("min_sample_length_before_truncation", 5)

        # -- Create Tensor Flow compute graph nodes
        self.tf_logical_predictions_per_timestep_per_batch = self._discriminator()
        (self.tf_discriminator_train_op,
         self.tf_discriminator_logical_loss_summary) = self._discriminator_optimizer()
        self._finish_init()

    def train(self, training_corpus, sample_grammar, train_test_split=None):
        self._train(
            self.tf_discriminator_train_op,
            [self.tf_discriminator_logical_loss_summary],
            training_corpus, sample_grammar, train_test_split,
            min_sample_length_before_truncation=self.min_sample_length_before_truncation)

    def name(self):
        return super().name()+"_fw{}_bw{}".format(
            "-".join(str(n) for n in self.fw_state_size_per_layer),
            "-".join(str(n) for n in self.bw_state_size_per_layer))

    def info(self):
        result = super().info()
        result["fw_state_size_per_layer"] = self.fw_state_size_per_layer
        result["bw_state_size_per_layer"] = self.bw_state_size_per_layer
        return result

    def discriminate(self, characters):
        pass

    # ----------------------[ Private Methods ]----------------------

    def _discriminator(self):
        with tf.name_scope("discriminator"):
            # -- Slice lexical features from lexical-logical input
            tf_lexical_embeddings_per_timestep_per_batch = self.tf_lexical_logical_embeddings_per_timestep_per_batch[
                :, :, :self.num_lexical_features]

            # -- Backward pass
            tf_discriminator_backward_cell = tf.contrib.rnn.MultiRNNCell([
                tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                self.bw_state_size_per_layer])
            tf_backward_embeddings_per_timestep_per_batch, _ = tf.nn.dynamic_rnn(
                cell=tf_discriminator_backward_cell,
                inputs=tf.reverse(tf_lexical_embeddings_per_timestep_per_batch, axis=[1]),
                initial_state=tf_discriminator_backward_cell.zero_state(
                    self.tf_lexical_logical_embeddings_per_timestep_per_batch_shape[0],
                    tf.float32),
                time_major=False)

            # -- Forward pass
            tf_discriminator_forward_cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                    self.fw_state_size_per_layer
                ]),
                self.num_logical_features)
            # -- Create a dynamically unrolled RNN to produce the character category discrimination
            tf_logical_predictions_per_timestep_per_batch, _ = tf.nn.dynamic_rnn(
                cell=tf_discriminator_forward_cell,
                inputs=tf.concat([
                    tf_lexical_embeddings_per_timestep_per_batch,
                    tf.reverse(tf_backward_embeddings_per_timestep_per_batch, axis=[1])], axis=2),
                sequence_length=self.tf_timesteps_per_batch,
                initial_state=tf_discriminator_forward_cell.zero_state(
                    self.tf_lexical_logical_embeddings_per_timestep_per_batch_shape[0],
                    tf.float32),
                time_major=False)

        return tf_logical_predictions_per_timestep_per_batch

    def _discriminator_optimizer(self):
        with tf.name_scope("discriminator_optimizer"):
            # -- Slice logical features from lexical-logical input
            tf_logical_embeddings_per_timestep_per_batch = self.tf_lexical_logical_embeddings_per_timestep_per_batch[
                :, :, -self.num_logical_features:]

            # -- Obtain global training step
            global_step = tf.contrib.framework.get_global_step()

            # -- Calculate the average cross entropy for the logical classes per timestep
            tf_logical_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf_logical_embeddings_per_timestep_per_batch,
                logits=self.tf_logical_predictions_per_timestep_per_batch,
                dim=2))

            # -- Create summaries for TensorBoard
            tf_logical_loss_summary = tf.summary.scalar("logical_loss", tf_logical_loss)

            # -- Define training op
            optimizer = tf.train.RMSPropOptimizer(self.tf_learning_rate)
            tf_train_op = tf.contrib.layers.optimize_loss(
                loss=tf_logical_loss,
                global_step=global_step,
                learning_rate=None,
                summaries=[],
                optimizer=optimizer)
        return tf_train_op, tf_logical_loss_summary
