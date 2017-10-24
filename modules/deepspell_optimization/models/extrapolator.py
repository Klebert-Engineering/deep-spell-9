# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import tensorflow as tf

from deepspell.models import extrapolator
from deepspell_optimization.models import optimizer


# ============================[ Local Imports ]==========================


# =======================[ LSTM Extrapolator Model ]=====================

class DSLstmExtrapolatorOptimizer(optimizer.DSModelOptimizerMixin, extrapolator.DSLstmExtrapolator):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class"""
        extrapolator.DSLstmExtrapolator.__init__(
            self,
            name_scope="extrapolator",
            version=3,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            args=kwargs)
        optimizer.DSModelOptimizerMixin.__init__(self, kwargs)

        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
            (self.tf_extrapolator_train_op,
             self.tf_extrapolator_logical_loss_summary,
             self.tf_extrapolator_lexical_loss_summary) = self._extrapolator_optimizer()

        self._finish_init_optimizer()

    def train(self, training_corpus, sample_grammar, train_test_split=None):
        self._train(
            self.tf_extrapolator_train_op,
            [self.tf_extrapolator_lexical_loss_summary, self.tf_extrapolator_logical_loss_summary],
            training_corpus, sample_grammar, train_test_split)

    def generate_name(self):
        return super().generate_name() + "_" + "-".join(str(n) for n in self.state_size_per_layer)

    def info(self):
        result = super().info()
        result["state_size_per_layer"] = self.state_size_per_layer
        return result

    # ----------------------[ Private Methods ]----------------------

    def _extrapolator_optimizer(self):
        with tf.name_scope("extrapolator_optimizer"):
            # -- Obtain global training step
            global_step = tf.contrib.framework.get_global_step()

            # -- Time indices are sliced as follows:
            #  For labels: First Input can be ignored
            #  For predictions: Last output (prediction after EOL) can be ignored
            #  For example:
            #   Label := a-b-c-d-e-a-.-0-0 -> Slice First -> b-c-d-e-a-.-0-0
            #   Pred. := b-c-d-e-a-.-0-0-0 -> Slice Last  -> b-c-d-e-a-.-0-0

            # -- Calculate the average cross entropy for the logical classes per timestep
            tf_logical_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                 labels=self.tf_lexical_logical_embeddings_per_timestep_per_batch[:, 1:, -self.num_logical_features:],
                 logits=self.tf_lexical_logical_predictions_per_timestep_per_batch[:, :-1, -self.num_logical_features:],
                 dim=2))

            # -- Calculate the average cross entropy for the lexical classes per timestep
            tf_lexical_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.tf_lexical_logical_embeddings_per_timestep_per_batch[:, 1:, :self.num_lexical_features],
                logits=self.tf_lexical_logical_predictions_per_timestep_per_batch[:, :-1, :self.num_lexical_features],
                dim=2))

            # -- Create summaries for TensorBoard
            tf_logical_loss_summary = tf.summary.scalar("logical_loss", tf_logical_loss)
            tf_lexical_loss_summary = tf.summary.scalar("lexical_loss", tf_lexical_loss)

            # -- Define training op
            optimizer = tf.train.RMSPropOptimizer(self.tf_learning_rate)
            tf_train_op = tf.contrib.layers.optimize_loss(
                loss=tf_logical_loss+tf_lexical_loss,
                global_step=global_step,
                learning_rate=None,
                summaries=[],
                optimizer=optimizer)
        return tf_train_op, tf_logical_loss_summary, tf_lexical_loss_summary
