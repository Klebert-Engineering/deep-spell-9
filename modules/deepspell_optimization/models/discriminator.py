# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import tensorflow as tf

# ============================[ Local Imports ]==========================

from deepspell.models import discriminator
from deepspell_optimization.models import optimizer


# ======================[ LSTM Discriminator Model ]=====================

class DSLstmDiscriminatorOptimizer(optimizer.DSModelOptimizerMixin, discriminator.DSLstmDiscriminator):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class
        :param total_features_per_character: The exact number of features per character,
         which is required for the Graph construction.
        """
        discriminator.DSLstmDiscriminator.__init__(
            self,
            name_scope="discriminator",
            version=3,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            args_to_update=kwargs)
        optimizer.DSModelOptimizerMixin.__init__(self, kwargs)

        # -- Read params
        self.min_sample_length_before_truncation = kwargs.pop("min_sample_length_before_truncation", 5)

        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
            (self.tf_discriminator_train_op,
             self.tf_discriminator_logical_loss_summary) = self._discriminator_optimizer()

        self._finish_init_optimizer()

    def train(self, training_corpus, sample_grammar, train_test_split=None):
        self._train(
            self.tf_discriminator_train_op,
            [self.tf_discriminator_logical_loss_summary],
            training_corpus, sample_grammar, train_test_split,
            min_sample_length_before_truncation=self.min_sample_length_before_truncation)

    def generate_name(self):
        return super().generate_name() + "_fw{}_bw{}".format(
            "-".join(str(n) for n in self.fw_state_size_per_layer),
            "-".join(str(n) for n in self.bw_state_size_per_layer))

    def info(self):
        result = super().info()
        result["fw_state_size_per_layer"] = self.fw_state_size_per_layer
        result["bw_state_size_per_layer"] = self.bw_state_size_per_layer
        return result

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
            tf_optimizer = tf.train.RMSPropOptimizer(self.tf_learning_rate)
            tf_train_op = tf.contrib.layers.optimize_loss(
                loss=tf_logical_loss,
                global_step=global_step,
                learning_rate=None,
                summaries=[],
                optimizer=tf_optimizer)
        return tf_train_op, tf_logical_loss_summary
