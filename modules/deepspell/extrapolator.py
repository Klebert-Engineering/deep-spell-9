# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import tensorflow as tf
import numpy as np

# ============================[ Local Imports ]==========================

from . import predictor
from . import corpus


# =======================[ LSTM Extrapolator Model ]=====================

class DSLstmExtrapolator(predictor.DSPredictor):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class"""
        super().__init__(
            name_scope="extrapolator",
            version=2,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            kwargs_to_update=kwargs)

        # -- Read params
        self.state_size_per_layer = kwargs.pop("state_size_per_layer", [128, 128])
        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
            (self.tf_extrapolator_cell,
             self.tf_lexical_logical_predictions_per_timestep_per_batch,
             self.tf_extrapolator_final_state_and_mem_stack) = self._extrapolator()
            (self.tf_maximum_stepwise_extrapolation_length,
             self.tf_stepwise_extrapolator_output) = self._stepwise_extrapolator()
            (self.tf_extrapolator_train_op,
             self.tf_extrapolator_logical_loss_summary,
             self.tf_extrapolator_lexical_loss_summary) = self._extrapolator_optimizer()
        self._finish_init()

    def train(self, training_corpus, sample_grammar, train_test_split=None):
        self._train(
            self.tf_extrapolator_train_op,
            [self.tf_extrapolator_lexical_loss_summary, self.tf_extrapolator_logical_loss_summary],
            training_corpus, sample_grammar, train_test_split)

    def name(self):
        return super().name()+"_"+"-".join(str(n) for n in self.state_size_per_layer)

    def info(self):
        result = super().info()
        result["state_size_per_layer"] = self.state_size_per_layer
        return result

    def extrapolate(self, completion_corpus, prefix_chars, prefix_classes, num_chars_to_predict):
        """
        Use this method to predict a postfix for the given prefix with this model.
        :param num_chars_to_predict: The number of characters to predict.
        :param completion_corpus: This corpus indicates the set of tokens that may be predicted, as well
         as the (char, embedding) mappings and terminal token classes.
        :param prefix_chars: The actual characters of the prefix to be completed.
        :param prefix_classes: The token classes of the characters in prefix_chars. This must be a coma-separated
         array that is exactly as long as `prefix_chars`. Each entry E_i must be the decimal numeric id of the
         class of character C_i.
        :return: A pair like
         (
            postfix_chars as per-timestep list of list of pairs like (char, probability),
            postfix_classes as per-timestep list of list of pairs like (class, probability)
         ),
         where len(postfix_classes) = len(postfix_chars) and len(postfix_classes) <= num_chars_to_predict.
         E.g. if num_chars_to_predict=2, charset={a,b,c}, classes={0,1,2}, a prediction may look like:

         ( [ [(a, .7),  [(b, .5)    [ [(1, .4),  [(2, .9)
              (b, .2),   (c, .4)       (0, .3),   (1, .1)
              (c, .1)],  (a, .1)] ],   (2, .3)],  (0, .0)] ] )
        """
        assert isinstance(completion_corpus, corpus.DSCorpus)
        assert len(prefix_chars) == len(prefix_classes)
        assert self.num_lexical_features == completion_corpus.num_lexical_features_per_character()
        assert self.num_logical_features == completion_corpus.num_logical_features_per_character()

        # -- Make sure to reshape the 2D timestep-features matrix into a 3D batch-timestep-features matrix
        embedded_prefix = completion_corpus.embed_characters(prefix_chars, prefix_classes)
        embedded_prefix_length = len(embedded_prefix)
        embedded_prefix = np.reshape(
            embedded_prefix,
            newshape=(1, embedded_prefix_length, self.num_logical_features + self.num_lexical_features))

        with self.graph.as_default():
            stepwise_extrapolator_output = self.session.run(self.tf_stepwise_extrapolator_output, feed_dict={
                self.tf_lexical_logical_embeddings_per_timestep_per_batch: embedded_prefix,
                self.tf_timesteps_per_batch: np.asarray([embedded_prefix_length]),
                self.tf_maximum_stepwise_extrapolation_length: num_chars_to_predict
            })
        assert len(stepwise_extrapolator_output) == num_chars_to_predict

        completion_chars = []
        completion_classes = []

        for prediction in stepwise_extrapolator_output:
            lexical_pd = sorted((  # sort char predictions by probability in descending order
                    (corpus.CHAR_SUBSET[i], p)
                    for i, p in enumerate(prediction[:self.num_lexical_features])
                ),
                key=lambda entry: entry[1],
                reverse=True)
            logical_pd = sorted((  # sort class predictions by probability in descending order
                    (completion_corpus.class_name_for_id(i) or "UNKNOWN_CLASS[{}]".format(i), p)
                    for i, p in enumerate(prediction[-self.num_logical_features:])
                ),
                key=lambda entry: entry[1],
                reverse=True)
            completion_chars.append(lexical_pd)
            completion_classes.append(logical_pd)

        return completion_chars, completion_classes

    # ----------------------[ Private Methods ]----------------------

    def _extrapolator(self):
        # -- Input placeholders: batch of training sequences and their lengths
        with tf.name_scope("extrapolator"):

            # -- LSTM cell for prediction
            tf_extrapolator_cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                    self.state_size_per_layer
                ]),
                self.num_logical_features + self.num_lexical_features
            )
            extrapolator_initial_state = tf_extrapolator_cell.zero_state(
                self.tf_lexical_logical_embeddings_per_timestep_per_batch_shape[0], tf.float32)

            # -- Create a dynamically unrolled RNN to produce the embedded document vector
            tf_lexical_logical_predictions_per_timestep_per_batch, tf_final_state = tf.nn.dynamic_rnn(
                cell=tf_extrapolator_cell,
                inputs=self.tf_lexical_logical_embeddings_per_timestep_per_batch,
                sequence_length=self.tf_timesteps_per_batch,
                initial_state=extrapolator_initial_state,
                time_major=False)

        return (
            tf_extrapolator_cell,
            tf_lexical_logical_predictions_per_timestep_per_batch,
            tf_final_state)

    def _stepwise_extrapolator(self):
        with tf.name_scope("stepwise_extrapolator"):
            tf_maximum_prediction_length = tf.placeholder(tf.int32)
            tf_stepwise_predictor_output = tf.TensorArray(dtype=tf.float32, size=tf_maximum_prediction_length)
            tf_initial_t = tf.constant(0, dtype=tf.int32)
            tf_initial_state = self.tf_extrapolator_final_state_and_mem_stack
            tf_initial_prev_output = self.tf_lexical_logical_embeddings_per_timestep_per_batch[:1, -1]

            def should_continue(t, *_):
                return t < tf_maximum_prediction_length

            def iteration(t, state, prev_output, predictions_per_timestep):
                with tf.variable_scope("rnn", reuse=True):
                    prev_output, state = self.tf_extrapolator_cell(
                        inputs=prev_output,
                        state=state)

                # -- Apply argmax/one-hot to cell output so rnn won't incrementally confuse itself.
                #  Also re-apply shape because otherwise tf.while_loop will
                #  complain about the shape of prev_output being unpredictable.
                one_hot_output = tf.reshape(tf.concat([
                        tf.one_hot(
                            tf.argmax(prev_output[:, :self.num_lexical_features], axis=1),
                            depth=self.num_lexical_features),
                        tf.one_hot(
                            tf.argmax(prev_output[:, -self.num_logical_features:], axis=1),
                            depth=self.num_logical_features),
                    ],
                    axis=1), shape=(1, self.num_lexical_features+self.num_logical_features))

                # -- Softmax and flatten prediction because only a single batch is actually predicted
                predictions_per_timestep = predictions_per_timestep.write(t, tf.reshape(tf.concat([
                    tf.nn.softmax(prev_output[:, :self.num_lexical_features], dim=1),
                    tf.nn.softmax(prev_output[:, -self.num_logical_features:], dim=1)],
                    axis=1), shape=(-1,)))
                return t + 1, state, one_hot_output, predictions_per_timestep

            _, _, _, tf_stepwise_predictor_output = tf.while_loop(
                should_continue, iteration,
                loop_vars=[tf_initial_t, tf_initial_state, tf_initial_prev_output, tf_stepwise_predictor_output])

        tf_stepwise_predictor_output = tf_stepwise_predictor_output.stack()
        return tf_maximum_prediction_length, tf_stepwise_predictor_output

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
