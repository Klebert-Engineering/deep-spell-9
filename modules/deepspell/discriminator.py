# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import tensorflow as tf
import numpy as np

# ============================[ Local Imports ]==========================

from . import predictor
from . import corpus
from . import featureset

# ======================[ LSTM Discriminator Model ]=====================

class DSLstmDiscriminator(predictor.DSPredictor):
    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class
        :param total_features_per_character: The exact number of features per character,
         which is required for the Graph construction.
        """
        super().__init__(
            name_scope="discriminator",
            version=3,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            kwargs_to_update=kwargs)

        # -- Read params
        self.fw_state_size_per_layer = kwargs.pop("fw_state_size_per_layer", [128, 128])
        self.bw_state_size_per_layer = kwargs.pop("bw_state_size_per_layer", [128, 128])
        self.min_sample_length_before_truncation = kwargs.pop("min_sample_length_before_truncation", 5)

        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
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
        return super().name() + "_fw{}_bw{}".format(
            "-".join(str(n) for n in self.fw_state_size_per_layer),
            "-".join(str(n) for n in self.bw_state_size_per_layer))

    def info(self):
        result = super().info()
        result["fw_state_size_per_layer"] = self.fw_state_size_per_layer
        result["bw_state_size_per_layer"] = self.bw_state_size_per_layer
        return result

    def discriminate(self, embedding_featureset, characters):
        """
        Use this method to predict the token classes for a given sequence of characters.
        :param embedding_featureset: This corpus indicates the set of tokens that may be predicted, as well
         as the (char, embedding) mappings and terminal token classes.
         It is important that the lexical/logical feature split of the given corpus matches exactly
         self.num_logical_features and self.num_lexical_features.
        :param characters: The characters whose classes should be predicted.
        :return: character_classes as per-timestep list of list of pairs like (class, probability),
         where len(character_classes) = len(characters) if characters ends in the corpus' EOL char,
         or len(character_classes) = len(characters) + 1 otherwise.
         E.g. if characters="xy", classes={0,1,2}, a prediction may look like:

         [ [(0, .7),  [(2, .5)   
            (2, .2),   (0, .4)   
            (1, .1)],  (1, .1)] ]
        """
        assert isinstance(embedding_featureset, featureset.DSFeatureSet)
        assert self.num_lexical_features == embedding_featureset.num_lexical_features()
        assert self.num_logical_features == embedding_featureset.num_logical_features()

        char_embeddings = embedding_featureset.embed_characters(characters)
        # -- Store the char emb. size, because they may be more than len(characters) due to EOL padding.
        char_embeddings_length = len(char_embeddings)
        # -- Make sure to reshape the 2D timestep-features matrix into a 3D batch-timestep-features matrix.
        char_embeddings = np.reshape(
            char_embeddings,
            newshape=(1, char_embeddings_length, self.num_lexical_features + self.num_logical_features))

        with self.graph.as_default():
            discriminator_output = self.session.run(self.tf_logical_predictions_per_timestep_per_batch, feed_dict={
                self.tf_lexical_logical_embeddings_per_timestep_per_batch: char_embeddings,
                self.tf_timesteps_per_batch: np.asarray([char_embeddings_length])
            })

        # -- Reshape 3D batch-timestep-features matrix back to 2D timestep-features matrix
        discriminator_output = np.reshape(
            discriminator_output,
            (char_embeddings_length, self.num_logical_features))

        # -- Apply softmax to output
        discriminator_output = np.exp(discriminator_output)
        discriminator_output /= np.reshape(np.sum(discriminator_output, axis=1), (len(discriminator_output), 1))

        # -- Sort output and translate classes to strings for convenience
        completion_classes = []
        for prediction in discriminator_output:
            logical_pd = sorted((  # sort class predictions by probability in descending order
                    (embedding_featureset.class_name_for_id(i) or "UNKNOWN_CLASS[{}]".format(i), float(p))
                    for i, p in enumerate(prediction)
                ),
                key=lambda entry: entry[1],
                reverse=True)
            completion_classes.append(logical_pd)

        return completion_classes

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
