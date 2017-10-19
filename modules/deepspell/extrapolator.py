# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import tensorflow as tf
import numpy as np

# ============================[ Local Imports ]==========================

from . import predictor
from . import featureset

# =======================[ LSTM Extrapolator Model ]=====================


class DSLstmExtrapolator(predictor.DSPredictor):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class"""
        super().__init__(
            name_scope="extrapolator",
            version=3,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            kwargs_to_update=kwargs)

        # -- Read params
        self.state_size_per_layer = kwargs.pop("state_size_per_layer", [128, 128])
        self.extrapolation_beam_count = kwargs.pop("extrapolation_beam_count", 5)

        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
            (self.tf_extrapolator_cell,
             self.tf_lexical_logical_predictions_per_timestep_per_batch,
             self.tf_extrapolator_final_state_tuple_stack) = self._extrapolator()
            (self.tf_maximum_stepwise_extrapolation_length,
             self.tf_eol_class_idx,
             self.tf_beam_probs,
             self.tf_stepwise_beam_output) = self._stepwise_beam_extrapolator()  # , self.tf_stepwise_debug_output
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

    def extrapolate(self, embedding_featureset, prefix_chars, prefix_classes, num_chars_to_predict):
        """
        Use this method to predict ranked postfixes for the given prefix with this model.
        :param num_chars_to_predict: The number of characters to predict.
        :param embedding_featureset: This corpus indicates the set of tokens that may be predicted, as well
         as the (char, embedding) mappings and terminal token classes.
        :param prefix_chars: The actual characters of the prefix to be completed.
        :param prefix_classes: The token classes of the characters in prefix_chars. This must be a coma-separated
         array that is exactly as long as `prefix_chars`. Each entry E_i must be the decimal numeric id of the
         class of character C_i.
        :return: A list of length self.num_beams like
         `[ [postfix_chars as str, postfix_classes as tuple, postfix_probability]+ ]`,
         where len(postfix_classes) = len(postfix_chars) and len(postfix_classes) <= num_chars_to_predict.
         E.g. if num_chars_to_predict=2, charset={a,b,c}, classes={0,1,2}, num_beams=2, a prediction may look like:

         [ ["aabb", [1, 1, 2, 2], .9)
           ["abab", [1, 1, 2, 2], .1) ]

         Note: A postfix length may stop short of num_chars_to_predict if it encounters EOL.
        """
        assert isinstance(embedding_featureset, featureset.DSFeatureSet)
        assert len(prefix_chars) == len(prefix_classes)
        assert self.num_lexical_features == embedding_featureset.num_lexical_features()
        assert self.num_logical_features == embedding_featureset.num_logical_features()

        # -- Make sure to reshape the 2D timestep-features matrix into a 3D batch-timestep-features matrix
        embedded_prefix = embedding_featureset.embed_characters(prefix_chars, prefix_classes, append_eol=False)
        embedded_prefix_length = len(embedded_prefix)
        embedded_prefix = np.reshape(
            embedded_prefix,
            newshape=(1, embedded_prefix_length, self.num_logical_features + self.num_lexical_features))

        with self.graph.as_default():
            stepwise_beam_output, beam_probs = self.session.run(  # , debug_output
                [self.tf_stepwise_beam_output, self.tf_beam_probs],  # , self.tf_stepwise_debug_output
                feed_dict={
                    self.tf_lexical_logical_embeddings_per_timestep_per_batch: embedded_prefix,
                    self.tf_timesteps_per_batch: np.asarray([embedded_prefix_length]),
                    self.tf_maximum_stepwise_extrapolation_length: num_chars_to_predict,
                    self.tf_eol_class_idx: self.featureset.eol_class_id
                })
        # print(debug_output)
        assert len(beam_probs) == self.extrapolation_beam_count
        assert np.shape(stepwise_beam_output)[1] == self.extrapolation_beam_count
        assert np.shape(stepwise_beam_output)[2] == 3  # prev_beam_id, char_id, class_id

        completions = [["", [], float(beam_probs[i])] for i in range(self.extrapolation_beam_count)]
        predecessor_beam_ids = list(range(self.extrapolation_beam_count))
        current_beam_step = np.shape(stepwise_beam_output)[0]

        # -- Decode beams from back to front
        while True:
            current_beam_step -= 1
            if current_beam_step < 0:
                break
            beam_step_data = stepwise_beam_output[current_beam_step]
            for i, (completion, predecessor_beam_id) in enumerate(zip(completions, predecessor_beam_ids)):
                predecessor_beam_ids[i], char_id, class_id = beam_step_data[predecessor_beam_id]
                class_id = self.featureset.class_name_for_id(class_id)
                if len(completion[1]) > 0 and class_id != completion[1][0]:
                    completion[0] = ""
                    completion[1] = []
                completion[0] = self.featureset.charset[char_id] + completion[0]
                completion[1] = [class_id] + completion[1]

        # -- Remove duplicate entries.
        unique_completions = dict()
        for completion in completions:
            if completion[0] not in unique_completions:
                unique_completions[completion[0]] = completion
            elif completion[2] > unique_completions[completion[0]][2]:
                unique_completions[completion[0]][2] = completion[2]

        # -- Sort beams by probability
        completions = sorted(unique_completions.values(), key=lambda completion: completion[2], reverse=True)
        return completions

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

    def _stepwise_beam_extrapolator(self):
        with tf.name_scope("stepwise_beam_extrapolator"):
            tf_maximum_prediction_length = tf.placeholder(tf.int32)
            tf_eol_class_idx = tf.placeholder(tf.int32)
            tf_stepwise_beam_output = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
            # tf_stepwise_debug_output = tf.TensorArray(dtype=tf.int32, size=tf_maximum_prediction_length-1)
            tf_beam_lexical_lookup_idx = tf.constant([
                (n, i)
                for n in range(self.extrapolation_beam_count)
                for i in range(self.num_lexical_features)], dtype=tf.int32)

            # -- Start at one, because first postfix character is predicted by the block extrapolator
            tf_initial_t = tf.constant(1, dtype=tf.int32)

            # -- The first prediction and lstm state come out of the block extrapolator,
            #  not the stepwise! The first prediction will be processed two-fold:
            #  * It will be arg-maxed/converted to one-hot so that it can be fed
            #    into the stepwise predictor.
            #  * It will be k-maxed and written as the first beam tails into tf_stepwise_beam_output[0]
            #    as the first extrapolated characters.
            tf_first_lexical_prediction = tf.nn.softmax(
                self.tf_lexical_logical_predictions_per_timestep_per_batch[0, -1, :self.num_lexical_features])
            tf_first_logical_class = tf.argmax(
                self.tf_lexical_logical_predictions_per_timestep_per_batch[0, -1, -self.num_logical_features:])
            tf_beam_state_stack = tuple(
                tf.contrib.rnn.LSTMStateTuple(
                    tf.tile(state_tuple.c, [self.extrapolation_beam_count, 1]),
                    tf.tile(state_tuple.h, [self.extrapolation_beam_count, 1])
                ) for state_tuple in self.tf_extrapolator_final_state_tuple_stack)
            tf_beam_probs, tf_beam_tails = tf.nn.top_k(tf_first_lexical_prediction, k=self.extrapolation_beam_count, sorted=False)
            tf_beam_tails = tf.concat([
                tf.reshape(tf.tile([0], [self.extrapolation_beam_count]), shape=(-1, 1)),  # Predecessor beam index for first step is irrelevant
                tf.reshape(tf_beam_tails, shape=(-1, 1)),  # top_k indices from first lexical prob. dist.
                tf.reshape(tf.tile([tf.cast(tf_first_logical_class, tf.int32)], [self.extrapolation_beam_count]), shape=(-1, 1))  # Always adapt best class for all beams
            ], axis=1)
            tf_stepwise_beam_output = tf_stepwise_beam_output.write(0, tf_beam_tails)
            tf_beam_probs = tf.log(tf_beam_probs)  # Current log-prob for each beam

            #  Per-beam per-step log-prob factor for each beam. Will be set to 0 when a beam encounters EOL.
            tf_unfinished_beams = tf.tile([True], [self.extrapolation_beam_count])

            def should_continue(t, beam_state_stack, beam_tails, beam_probs, stepwise_beam_output, unfinished_beams):
                return tf.logical_and(t < tf_maximum_prediction_length, tf.count_nonzero(unfinished_beams) > 0)

            def iteration(t, beam_state_stack, beam_tails, beam_probs, stepwise_beam_output, unfinished_beams):  # , debug_output
                # -- Prepare information that allows for only furthering unfinished beams
                num_unfinished_beams = tf.cast(tf.count_nonzero(unfinished_beams), tf.int32)
                num_finished_beams = self.extrapolation_beam_count - num_unfinished_beams
                _, beam_indices_sorted_by_finished = tf.nn.top_k(
                    tf.cast(unfinished_beams, tf.int8), sorted=True, k=self.extrapolation_beam_count)
                finished_beam_indices = beam_indices_sorted_by_finished[num_unfinished_beams:]
                unfinished_beam_indices = beam_indices_sorted_by_finished[:num_unfinished_beams]
                unfinished_beam_tails = tf.gather(beam_tails, unfinished_beam_indices)
                unfinished_beam_probs = tf.gather(beam_probs, unfinished_beam_indices)
                unfinished_beam_lstm_states = tuple(
                    tf.contrib.rnn.LSTMStateTuple(
                        tf.gather(state_tuple.c, unfinished_beam_indices),
                        tf.gather(state_tuple.h, unfinished_beam_indices)
                    ) for state_tuple in beam_state_stack)

                # -- Get beam predictions and new lstm states
                with tf.variable_scope("rnn", reuse=True):
                    lexical_emb = tf.one_hot(unfinished_beam_tails[:, 1], depth=self.num_lexical_features)
                    logical_emb = tf.one_hot(unfinished_beam_tails[:, 2], depth=self.num_logical_features)
                    beam_predictions, beam_state_stack = self.tf_extrapolator_cell(
                        state=unfinished_beam_lstm_states,
                        inputs=tf.concat([lexical_emb, logical_emb], axis=1))

                # -- Extract 2D lexical/logical embs. from pred., Argmax logical predictions
                lexical_beam_pred = tf.nn.softmax(beam_predictions[:, :self.num_lexical_features])
                logical_beam_pred = tf.nn.softmax(beam_predictions[:, -self.num_logical_features:])
                logical_beam_pred = tf.cast(tf.argmax(logical_beam_pred, axis=1), tf.int32)

                # -- Flatten and k-max lexical beam predictions
                lexical_beam_pred = tf.log(lexical_beam_pred) + tf.reshape(unfinished_beam_probs, shape=(-1, 1))
                lexical_beam_pred = tf.reshape(lexical_beam_pred, shape=(-1,))
                unfinished_beam_probs, top_lexical_beam_pred_ids = tf.nn.top_k(lexical_beam_pred, k=num_unfinished_beams, sorted=False)

                # -- Adapt new probability values for the beams
                beam_probs = tf.reshape(tf.concat([
                    unfinished_beam_probs,
                    tf.gather(beam_probs, finished_beam_indices)
                ], axis=0), shape=(self.extrapolation_beam_count,))

                # -- Gather new beam tail index values.
                #  Note, that these beam ids are local to the unfinished beam indices! They will therefore
                #  be translated to global beam indices via a <gather-lookup> after the new beam tails are processed.
                top_lexical_beam_pred_ids = tf.gather(tf_beam_lexical_lookup_idx, top_lexical_beam_pred_ids)
                top_beam_pred_ids = top_lexical_beam_pred_ids[:, 0]
                # debug_output = debug_output.write(t-1, top_beam_pred_ids)
                top_logical_beam_pred_ids = tf.gather(logical_beam_pred, top_beam_pred_ids)
                beam_tails = tf.reshape(tf.concat([
                    tf.reshape(tf.concat([  # |--> Aforementioned <gather-lookup>
                        tf.gather(unfinished_beam_indices, top_beam_pred_ids),
                        finished_beam_indices], axis=0), shape=(-1, 1)),
                    tf.reshape(tf.concat([
                        top_lexical_beam_pred_ids[:, 1],
                        tf.tile([0], [num_finished_beams])], axis=0), shape=(-1, 1)),
                    tf.reshape(tf.concat([
                        top_logical_beam_pred_ids,
                        tf.tile([0], [num_finished_beams])], axis=0), shape=(-1, 1))
                ], axis=1), shape=(self.extrapolation_beam_count, 3))

                # -- Check and note which beams just finished (beam class switched from original)
                unfinished_beams = tf.reshape(tf.concat([
                    tf.equal(top_logical_beam_pred_ids, unfinished_beam_tails[:, 2]),
                    tf.zeros([num_finished_beams], dtype=tf.bool)
                ], axis=0), shape=(self.extrapolation_beam_count,))

                # -- Gather new LSTM states. Make sure that exactly extrapolation_beam_count states are written
                #  per state stack component, such that their shape is invariant.
                padded_top_beam_pred_ids = tf.reshape(tf.concat([
                    top_beam_pred_ids,
                    tf.tile([0], [num_finished_beams])
                ], axis=0), shape=(self.extrapolation_beam_count,))
                beam_state_stack = tuple(
                    tf.contrib.rnn.LSTMStateTuple(
                        tf.gather(state_tuple.c, padded_top_beam_pred_ids),
                        tf.gather(state_tuple.h, padded_top_beam_pred_ids)
                    ) for state_tuple in beam_state_stack)

                stepwise_beam_output = stepwise_beam_output.write(t, beam_tails)
                t = t + 1
                return (
                    t,
                    beam_state_stack,
                    beam_tails,
                    beam_probs,
                    stepwise_beam_output,
                    unfinished_beams)  # , debug_output

            _, _, _, tf_beam_probs, tf_stepwise_beam_output, _ = tf.while_loop(  # , tf_stepwise_debug_output
                should_continue, iteration,
                back_prop=False,
                loop_vars=[
                    tf_initial_t,
                    tf_beam_state_stack,
                    tf_beam_tails,
                    tf_beam_probs,
                    tf_stepwise_beam_output,
                    tf_unfinished_beams])  # , tf_stepwise_debug_output

            tf_stepwise_beam_output = tf_stepwise_beam_output.stack()
            # tf_stepwise_debug_output = tf_stepwise_debug_output.stack()

        return (
            tf_maximum_prediction_length,
            tf_eol_class_idx,
            tf_beam_probs,
            tf_stepwise_beam_output)  # , tf_stepwise_debug_output

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
