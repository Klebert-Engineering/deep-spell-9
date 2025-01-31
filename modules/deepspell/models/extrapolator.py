# (C) 2018-present Klebert Engineering

# ===============================[ Imports ]=============================

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ============================[ Local Imports ]==========================

from deepspell import featureset
from deepspell.models import modelbase

# =======================[ Custom RNN Cells ]=======================

class KerasLSTMCell:
    """A basic LSTM cell implementation compatible with TF1.x dynamic_rnn."""
    
    def __init__(self, units, cell_idx=0):
        self._num_units = units
        self._cell_idx = cell_idx

    @property
    def state_size(self):
        return (self._num_units, self._num_units)  # (c, h)

    @property
    def output_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        c = tf.zeros([batch_size, self._num_units], dtype=dtype)
        h = tf.zeros([batch_size, self._num_units], dtype=dtype)
        return (c, h)

    def __call__(self, inputs, state=None, states=None, scope=None):
        state = state if state is not None else states
        with tf.variable_scope("basic_lstm_cell"):
            c, h = state
            
            # Get the concatenated input
            concat = tf.concat([inputs, h], axis=1)
            input_dim = concat.shape[-1]
            
            # Create variables if they don't exist
            kernel = tf.get_variable(
                "kernel",
                shape=[input_dim, self._num_units * 4],
                initializer=tf.glorot_uniform_initializer()
            )
            bias = tf.get_variable(
                "bias",
                shape=[self._num_units * 4],
                initializer=tf.zeros_initializer()
            )
            
            # Apply weights
            gates = tf.matmul(concat, kernel) + bias
            
            # Split gates
            i, j, f, o = tf.split(gates, 4, axis=1)
            
            # Apply activations
            i = tf.sigmoid(i)  # input gate
            j = tf.tanh(j)     # new input
            f = tf.sigmoid(f)  # forget gate
            o = tf.sigmoid(o)  # output gate
            
            # Compute new cell state
            new_c = f * c + i * j
            
            # Compute new hidden state
            new_h = o * tf.tanh(new_c)
            
            return new_h, (new_c, new_h)

class MultiRNNCell:
    """A multi-layer RNN cell implementation compatible with TF1.x dynamic_rnn."""
    
    def __init__(self, num_units, num_layers):
        self._cells = [
            KerasLSTMCell(units=num_units, cell_idx=i)
            for i in range(num_layers)
        ]

    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def zero_state(self, batch_size, dtype):
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)

    def __call__(self, inputs, state=None, states=None, scope=None):
        state = state if state is not None else states
        cur_inp = inputs
        new_states = []
        for i, (cell, cell_state) in enumerate(zip(self._cells, state)):
            with tf.variable_scope(f"cell_{i}"):
                cur_inp, new_state = cell(cur_inp, cell_state)
                new_states.append(new_state)
        return cur_inp, tuple(new_states)

class OutputProjectionWrapper:
    """An output projection wrapper compatible with TF1.x dynamic_rnn."""
    
    def __init__(self, cell, output_size):
        self._cell = cell
        self._output_size_value = output_size

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size_value

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state=None, states=None, scope=None):
        state = state if state is not None else states
        with tf.variable_scope("output_projection_wrapper"):
            with tf.variable_scope("multi_rnn_cell"):
                output, new_state = self._cell(inputs, state)
            
            # Create projection variables
            kernel = tf.get_variable(
                "kernel",
                shape=[self._cell.output_size, self._output_size_value],
                initializer=tf.glorot_uniform_initializer()
            )
            bias = tf.get_variable(
                "bias",
                shape=[self._output_size_value],
                initializer=tf.zeros_initializer()
            )
            
            projected_output = tf.matmul(output, kernel) + bias
            return projected_output, new_state

def create_rnn_cell(num_units, num_layers, output_size):
    """Creates a multi-layer RNN cell with output projection."""
    with tf.variable_scope("rnn"):
        multi_cell = MultiRNNCell(num_units, num_layers)
        return OutputProjectionWrapper(multi_cell, output_size)

# =======================[ LSTM Extrapolator Model ]=====================


class DSLstmExtrapolator(modelbase.DSModelBase):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", args_to_update=None, **kwargs):
        """Documentation in base Model class"""

        if not args_to_update:
            args_to_update = dict()
        args_to_update.update(kwargs)

        super().__init__(
            name_scope="extrapolator",
            version=3,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            args_to_update=args_to_update)

        # -- Read params
        self.state_size_per_layer = args_to_update.pop("state_size_per_layer", [128, 128])
        self.extrapolation_beam_count = args_to_update.pop("extrapolation_beam_count", 5)

        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
            (self.tf_extrapolator_cell,
             self.tf_lexical_logical_predictions_per_timestep_per_batch,
             self.tf_extrapolator_final_state_tuple_stack) = self._extrapolator()
            (self.tf_maximum_stepwise_extrapolation_length,
             self.tf_eol_class_idx,
             self.tf_beam_probs,
             self.tf_stepwise_beam_output) = self._stepwise_beam_extrapolator()  # , self.tf_stepwise_debug_output
        self._finish_init_base()

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
            num_units = self.state_size_per_layer[0]  # All layers have same size
            num_layers = len(self.state_size_per_layer)
            
            multi_cell = MultiRNNCell(num_units, num_layers)
            tf_extrapolator_cell = OutputProjectionWrapper(
                multi_cell,
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
                dtype=tf.float32,
                time_major=False,
                scope="rnn")

        return (
            tf_extrapolator_cell,
            tf_lexical_logical_predictions_per_timestep_per_batch,
            tf_final_state)

    def _stepwise_beam_extrapolator(self):
        with tf.name_scope("stepwise_beam_extrapolator"):
            tf_maximum_prediction_length = tf.placeholder(tf.int32)
            tf_eol_class_idx = tf.placeholder(tf.int32)
            tf_stepwise_beam_output = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
            tf_beam_lexical_lookup_idx = tf.constant([
                (n, i)
                for n in range(self.extrapolation_beam_count)
                for i in range(self.num_lexical_features)], dtype=tf.int32)

            # -- Start at one, because first postfix character is predicted by the block extrapolator
            tf_initial_t = tf.constant(1, dtype=tf.int32)

            # -- The first prediction and lstm state come out of the block extrapolator
            tf_first_lexical_prediction = tf.nn.softmax(
                self.tf_lexical_logical_predictions_per_timestep_per_batch[0, -1, :self.num_lexical_features])
            tf_first_logical_class = tf.argmax(
                self.tf_lexical_logical_predictions_per_timestep_per_batch[0, -1, -self.num_logical_features:])
            
            # Create initial beam states by tiling the final states
            tf_beam_state_stack = []
            for state_tuple in self.tf_extrapolator_final_state_tuple_stack:
                h = tf.tile(state_tuple[0], [self.extrapolation_beam_count, 1])
                c = tf.tile(state_tuple[1], [self.extrapolation_beam_count, 1])
                tf_beam_state_stack.append((h, c))
            tf_beam_state_stack = tuple(tf_beam_state_stack)
            
            tf_beam_probs, tf_beam_tails = tf.nn.top_k(tf_first_lexical_prediction, k=self.extrapolation_beam_count, sorted=False)
            tf_beam_tails = tf.concat([
                tf.reshape(tf.tile([0], [self.extrapolation_beam_count]), shape=(-1, 1)),
                tf.reshape(tf_beam_tails, shape=(-1, 1)),
                tf.reshape(tf.tile([tf.cast(tf_first_logical_class, tf.int32)], [self.extrapolation_beam_count]), shape=(-1, 1))
            ], axis=1)
            tf_stepwise_beam_output = tf_stepwise_beam_output.write(0, tf_beam_tails)
            tf_beam_probs = tf.log(tf_beam_probs)

            tf_unfinished_beams = tf.tile([True], [self.extrapolation_beam_count])

            def should_continue(t, beam_state_stack, beam_tails, beam_probs, stepwise_beam_output, unfinished_beams):
                return tf.logical_and(t < tf_maximum_prediction_length, tf.count_nonzero(unfinished_beams) > 0)

            def iteration(t, beam_state_stack, beam_tails, beam_probs, stepwise_beam_output, unfinished_beams):
                num_unfinished_beams = tf.cast(tf.count_nonzero(unfinished_beams), tf.int32)
                num_finished_beams = self.extrapolation_beam_count - num_unfinished_beams
                _, beam_indices_sorted_by_finished = tf.nn.top_k(
                    tf.cast(unfinished_beams, tf.int8), sorted=True, k=self.extrapolation_beam_count)
                finished_beam_indices = beam_indices_sorted_by_finished[num_unfinished_beams:]
                unfinished_beam_indices = beam_indices_sorted_by_finished[:num_unfinished_beams]
                unfinished_beam_tails = tf.gather(beam_tails, unfinished_beam_indices)
                unfinished_beam_probs = tf.gather(beam_probs, unfinished_beam_indices)
                
                # Update LSTM states for unfinished beams
                unfinished_beam_lstm_states = []
                for state_tuple in beam_state_stack:
                    h = tf.gather(state_tuple[0], unfinished_beam_indices)
                    c = tf.gather(state_tuple[1], unfinished_beam_indices)
                    unfinished_beam_lstm_states.append((h, c))
                unfinished_beam_lstm_states = tuple(unfinished_beam_lstm_states)

                # Get beam predictions and new lstm states
                with tf.variable_scope("rnn", reuse=True):
                    lexical_emb = tf.one_hot(unfinished_beam_tails[:, 1], depth=self.num_lexical_features)
                    logical_emb = tf.one_hot(unfinished_beam_tails[:, 2], depth=self.num_logical_features)
                    beam_predictions, beam_state_stack = self.tf_extrapolator_cell(
                        inputs=tf.concat([lexical_emb, logical_emb], axis=1),
                        states=unfinished_beam_lstm_states)

                # Extract predictions and process them
                lexical_beam_pred = tf.nn.softmax(beam_predictions[:, :self.num_lexical_features])
                logical_beam_pred = tf.nn.softmax(beam_predictions[:, -self.num_logical_features:])
                logical_beam_pred = tf.cast(tf.argmax(logical_beam_pred, axis=1), tf.int32)

                lexical_beam_pred = tf.log(lexical_beam_pred) + tf.reshape(unfinished_beam_probs, shape=(-1, 1))
                lexical_beam_pred = tf.reshape(lexical_beam_pred, shape=(-1,))
                unfinished_beam_probs, top_lexical_beam_pred_ids = tf.nn.top_k(
                    lexical_beam_pred, k=num_unfinished_beams, sorted=False)

                # Update beam probabilities
                beam_probs = tf.reshape(tf.concat([
                    unfinished_beam_probs,
                    tf.gather(beam_probs, finished_beam_indices)
                ], axis=0), shape=(self.extrapolation_beam_count,))

                # Process beam predictions
                top_lexical_beam_pred_ids = tf.gather(tf_beam_lexical_lookup_idx, top_lexical_beam_pred_ids)
                top_beam_pred_ids = top_lexical_beam_pred_ids[:, 0]
                top_logical_beam_pred_ids = tf.gather(logical_beam_pred, top_beam_pred_ids)
                
                # Update beam tails
                beam_tails = tf.reshape(tf.concat([
                    tf.reshape(tf.concat([
                        tf.gather(unfinished_beam_indices, top_beam_pred_ids),
                        finished_beam_indices], axis=0), shape=(-1, 1)),
                    tf.reshape(tf.concat([
                        top_lexical_beam_pred_ids[:, 1],
                        tf.tile([0], [num_finished_beams])], axis=0), shape=(-1, 1)),
                    tf.reshape(tf.concat([
                        top_logical_beam_pred_ids,
                        tf.tile([0], [num_finished_beams])], axis=0), shape=(-1, 1))
                ], axis=1), shape=(self.extrapolation_beam_count, 3))

                # Update unfinished beams status
                unfinished_beams = tf.reshape(tf.concat([
                    tf.equal(top_logical_beam_pred_ids, unfinished_beam_tails[:, 2]),
                    tf.zeros([num_finished_beams], dtype=tf.bool)
                ], axis=0), shape=(self.extrapolation_beam_count,))

                # Update LSTM states
                padded_top_beam_pred_ids = tf.reshape(tf.concat([
                    top_beam_pred_ids,
                    tf.tile([0], [num_finished_beams])
                ], axis=0), shape=(self.extrapolation_beam_count,))
                
                # Update all states in the stack
                new_beam_state_stack = []
                for state_tuple in beam_state_stack:
                    h = tf.gather(state_tuple[0], padded_top_beam_pred_ids)
                    c = tf.gather(state_tuple[1], padded_top_beam_pred_ids)
                    new_beam_state_stack.append((h, c))
                beam_state_stack = tuple(new_beam_state_stack)

                stepwise_beam_output = stepwise_beam_output.write(t, beam_tails)
                t = t + 1
                return (t, beam_state_stack, beam_tails, beam_probs, stepwise_beam_output, unfinished_beams)

            _, _, _, tf_beam_probs, tf_stepwise_beam_output, _ = tf.while_loop(
                should_continue, iteration,
                back_prop=False,
                loop_vars=[
                    tf_initial_t,
                    tf_beam_state_stack,
                    tf_beam_tails,
                    tf_beam_probs,
                    tf_stepwise_beam_output,
                    tf_unfinished_beams])

            tf_stepwise_beam_output = tf_stepwise_beam_output.stack()

        return (
            tf_maximum_prediction_length,
            tf_eol_class_idx,
            tf_beam_probs,
            tf_stepwise_beam_output)

    def stepwise_beam_extrapolate(self, session, input_features, beam_size, max_steps):
        # Initialize beam states
        initial_state = session.run(self.extrapolator_initial_state)
        beam_states = [(1.0, [], initial_state)]  # (prob, sequence, state)
        
        for step in range(max_steps):
            # Get all current beams
            all_beam_probs = []
            all_beam_tails = []
            all_beam_states = []
            
            # For each beam, get predictions for next step
            for beam_prob, beam_seq, beam_state in beam_states:
                if len(beam_seq) == 0:
                    current_input = input_features
                else:
                    current_input = beam_seq[-1]
                
                # Run one step prediction
                feed_dict = {
                    self.input_features: [current_input],
                    self.extrapolator_initial_state: beam_state
                }
                predictions, final_state = session.run(
                    [self.extrapolator_predictions, self.extrapolator_final_state],
                    feed_dict=feed_dict
                )
                
                # Get top k predictions
                top_k_probs, top_k_indices = tf.nn.top_k(
                    tf.nn.softmax(predictions[0]), k=beam_size
                )
                top_k_probs = top_k_probs.numpy()
                top_k_indices = top_k_indices.numpy()
                
                # Add each prediction to beam candidates
                for prob, idx in zip(top_k_probs, top_k_indices):
                    all_beam_probs.append(beam_prob * prob)
                    all_beam_tails.append(beam_seq + [idx])
                    all_beam_states.append(final_state)
            
            # Select top beams
            beam_indices = np.argsort(all_beam_probs)[-beam_size:]
            beam_states = [
                (all_beam_probs[i], all_beam_tails[i], all_beam_states[i])
                for i in beam_indices
            ]
            
            # Check if all beams have reached max length
            if all(len(seq) >= max_steps for _, seq, _ in beam_states):
                break
        
        return beam_states
