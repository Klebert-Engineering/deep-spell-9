# (C) 2018-present Klebert Engineering

# ===============================[ Imports ]=============================

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
from collections import defaultdict

# ============================[ Local Imports ]==========================

from deepspell import featureset
from deepspell.models import modelbase
from deepspell.models.extrapolator import KerasLSTMCell, MultiRNNCell, OutputProjectionWrapper


# ======================[ LSTM Discriminator Model ]=====================

class DSLstmDiscriminator(modelbase.DSModelBase):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", args_to_update=None, **kwargs):
        """Documentation in base Model class"""

        if not args_to_update:
            args_to_update = dict()
        args_to_update.update(kwargs)

        super().__init__(
            name_scope="discriminator",
            version=3,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            args_to_update=args_to_update)

        # -- Read params
        self.fw_state_size_per_layer = args_to_update.pop("fw_state_size_per_layer", [128, 128])
        self.bw_state_size_per_layer = args_to_update.pop("bw_state_size_per_layer", [128, 128])

        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
            self.tf_logical_predictions_per_timestep_per_batch = self._discriminator()
        self._finish_init_base()

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
            tf_lexical_embeddings_per_timestep_per_batch = \
                self.tf_lexical_logical_embeddings_per_timestep_per_batch[:, :, :self.num_lexical_features]

            with tf.variable_scope("rnn"):
                # -- Create backward LSTM cell
                with tf.variable_scope("multi_rnn_cell"):
                    with tf.variable_scope("cell_0"):
                        with tf.variable_scope("basic_lstm_cell"):
                            # Get input shape
                            batch_size = tf.shape(tf_lexical_embeddings_per_timestep_per_batch)[0]
                            max_time = tf.shape(tf_lexical_embeddings_per_timestep_per_batch)[1]
                            
                            # Create variables
                            input_size = self.num_lexical_features
                            hidden_size = self.bw_state_size_per_layer[0]
                            kernel = tf.get_variable(
                                "kernel",
                                shape=[input_size + hidden_size, hidden_size * 4],
                                initializer=tf.glorot_uniform_initializer()
                            )
                            bias = tf.get_variable(
                                "bias",
                                shape=[hidden_size * 4],
                                initializer=tf.zeros_initializer()
                            )
                            
                            # Initialize state
                            c = tf.zeros([batch_size, hidden_size], dtype=tf.float32)
                            h = tf.zeros([batch_size, hidden_size], dtype=tf.float32)
                            
                            # Create TensorArray for outputs
                            outputs_array = tf.TensorArray(
                                dtype=tf.float32,
                                size=max_time,
                                element_shape=[None, hidden_size]
                            )
                            
                            # Define loop body
                            def body(t, c, h, outputs_array):
                                # Get input for this timestep
                                current_input = tf_lexical_embeddings_per_timestep_per_batch[:, t, :]
                                
                                # Concatenate input and previous hidden state
                                concat = tf.concat([current_input, h], axis=1)
                                
                                # Apply weights
                                gates = tf.matmul(concat, kernel) + bias
                                
                                # Split gates
                                i, j, f, o = tf.split(gates, 4, axis=1)
                                
                                # Apply activations
                                i = tf.sigmoid(i)  # input gate
                                j = tf.tanh(j)     # new input
                                f = tf.sigmoid(f)  # forget gate
                                o = tf.sigmoid(o)  # output gate
                                
                                # Update cell state
                                new_c = f * c + i * j
                                
                                # Update hidden state
                                new_h = o * tf.tanh(new_c)
                                
                                # Write output
                                outputs_array = outputs_array.write(t, new_h)
                                
                                return t + 1, new_c, new_h, outputs_array
                            
                            # Define loop condition
                            def cond(t, c, h, outputs_array):
                                return t < max_time
                            
                            # Run the loop
                            _, _, _, outputs_array = tf.while_loop(
                                cond,
                                body,
                                loop_vars=[
                                    tf.constant(0),
                                    c,
                                    h,
                                    outputs_array
                                ]
                            )
                            
                            # Stack outputs
                            tf_backward_embeddings_per_timestep_per_batch = outputs_array.stack()
                            tf_backward_embeddings_per_timestep_per_batch = tf.transpose(
                                tf_backward_embeddings_per_timestep_per_batch, [1, 0, 2])

                # -- Create forward LSTM cell
                with tf.variable_scope("output_projection_wrapper"):
                    # Concatenate forward and backward inputs
                    combined_input = tf.concat([
                        tf_lexical_embeddings_per_timestep_per_batch,
                        tf_backward_embeddings_per_timestep_per_batch], axis=2)
                    
                    with tf.variable_scope("multi_rnn_cell"):
                        # Process each layer
                        current_input = combined_input
                        for i in range(len(self.fw_state_size_per_layer)):
                            with tf.variable_scope(f"cell_{i}"):
                                with tf.variable_scope("basic_lstm_cell"):
                                    # Get layer sizes
                                    if i == 0:
                                        input_size = combined_input.shape[-1]
                                    else:
                                        input_size = self.fw_state_size_per_layer[i-1]
                                    hidden_size = self.fw_state_size_per_layer[i]
                                    
                                    # Create variables
                                    kernel = tf.get_variable(
                                        "kernel",
                                        shape=[input_size + hidden_size, hidden_size * 4],
                                        initializer=tf.glorot_uniform_initializer()
                                    )
                                    bias = tf.get_variable(
                                        "bias",
                                        shape=[hidden_size * 4],
                                        initializer=tf.zeros_initializer()
                                    )
                                    
                                    # Initialize state
                                    c = tf.zeros([batch_size, hidden_size], dtype=tf.float32)
                                    h = tf.zeros([batch_size, hidden_size], dtype=tf.float32)
                                    
                                    # Create TensorArray for outputs
                                    outputs_array = tf.TensorArray(
                                        dtype=tf.float32,
                                        size=max_time,
                                        element_shape=[None, hidden_size]
                                    )
                                    
                                    # Define loop body
                                    def body(t, c, h, outputs_array):
                                        # Get input for this timestep
                                        timestep_input = current_input[:, t, :]
                                        
                                        # Concatenate input and previous hidden state
                                        concat = tf.concat([timestep_input, h], axis=1)
                                        
                                        # Apply weights
                                        gates = tf.matmul(concat, kernel) + bias
                                        
                                        # Split gates
                                        i, j, f, o = tf.split(gates, 4, axis=1)
                                        
                                        # Apply activations
                                        i = tf.sigmoid(i)  # input gate
                                        j = tf.tanh(j)     # new input
                                        f = tf.sigmoid(f)  # forget gate
                                        o = tf.sigmoid(o)  # output gate
                                        
                                        # Update cell state
                                        new_c = f * c + i * j
                                        
                                        # Update hidden state
                                        new_h = o * tf.tanh(new_c)
                                        
                                        # Write output
                                        outputs_array = outputs_array.write(t, new_h)
                                        
                                        return t + 1, new_c, new_h, outputs_array
                                    
                                    # Define loop condition
                                    def cond(t, c, h, outputs_array):
                                        return t < max_time
                                    
                                    # Run the loop
                                    _, _, _, outputs_array = tf.while_loop(
                                        cond,
                                        body,
                                        loop_vars=[
                                            tf.constant(0),
                                            c,
                                            h,
                                            outputs_array
                                        ]
                                    )
                                    
                                    # Stack outputs
                                    current_input = outputs_array.stack()
                                    current_input = tf.transpose(current_input, [1, 0, 2])

                    # Create projection variables
                    kernel = tf.get_variable(
                        "kernel",
                        shape=[self.fw_state_size_per_layer[-1], self.num_logical_features],
                        initializer=tf.glorot_uniform_initializer()
                    )
                    bias = tf.get_variable(
                        "bias",
                        shape=[self.num_logical_features],
                        initializer=tf.zeros_initializer()
                    )

                    # Reshape to 2D for matmul
                    output_reshaped = tf.reshape(current_input, [-1, self.fw_state_size_per_layer[-1]])
                    
                    # Apply projection
                    projected = tf.matmul(output_reshaped, kernel) + bias
                    
                    # Reshape back to 3D
                    tf_logical_predictions_per_timestep_per_batch = tf.reshape(
                        projected, [batch_size, max_time, self.num_logical_features])

            return tf_logical_predictions_per_timestep_per_batch


# ======================[ Tokenization Utility ]=====================

def tokenize_class_annotated_characters(characters, class_annotation_per_character, separator_fn=lambda ch: ch in " -,"):
    """
    Tokenizes a string based on class distribution per character into a set of per-class tokens.
    The smallest token units are strings of characters where for no character except the first
    the separator_fn predicate returns true. Token unit class appropriation is determined based
    on the argmax of the character-wise cumulative neg-log-prob of the token unit for a given class.
    :param characters: The string of characters which should be tokenized.
    :param class_annotation_per_character: Output from DSLstmDiscriminator.discriminate(..., characters).
    :param separator_fn:
    :return: A map like {<classname>: <concatenated tokens>}. The <classname> keys are entirely drawn from
     the class names mentioned in @class_annotation_per_character.
    """
    assert separator_fn
    result_map = defaultdict(str)
    current_token = ""
    current_logprob_per_class = defaultdict(float)
    for ch, classname_prob_list in zip(characters, class_annotation_per_character):
        if separator_fn(ch) and current_logprob_per_class and current_token:
            result_map[max(current_logprob_per_class, key=lambda cl: current_logprob_per_class[cl])] += current_token
            current_token = ""
            current_logprob_per_class.clear()
        current_token += ch
        for classname, prob in classname_prob_list:
            current_logprob_per_class[classname] += math.log(prob+1e-9)
    # -- Make sure to add the last token too
    if current_logprob_per_class and current_token:
        result_map[max(current_logprob_per_class, key=lambda cl: current_logprob_per_class[cl])] += current_token
    # -- Trim all entries
    result_map = {key: value.strip() for key, value in result_map.items()}
    return result_map


def extract_best_class_sequence(characters, class_annotation_per_character, separator_fn=lambda ch: ch in " -,"):
    """
    Returns best class combination based on class distribution per character into list of classes.
    The smallest token units are strings of characters where for no character except the first
    the separator_fn predicate returns true. Token unit class appropriation is determined based
    on the argmax of the character-wise cumulative log-prob of the token unit for a given class.
    :param characters: The string of characters which should be tokenized.
    :param class_annotation_per_character: Output from DSLstmDiscriminator.discriminate(..., characters).
    :param separator_fn:
    :return: A map like {<classname>: <concatenated tokens>}. The <classname> keys are entirely drawn from
     the class names mentioned in @class_annotation_per_character.
    """
    assert separator_fn
    result_sequence = []
    current_token_length = 0
    current_logprob_per_class = defaultdict(float)
    for ch, classname_prob_list in zip(characters, class_annotation_per_character):
        if separator_fn(ch) and current_logprob_per_class and current_token_length:
            result_sequence += \
                current_token_length*[max(current_logprob_per_class, key=lambda cl: current_logprob_per_class[cl])]
            current_token_length = 0
            current_logprob_per_class.clear()
        current_token_length += 1
        for classname, prob in classname_prob_list:
            current_logprob_per_class[classname] += math.log(prob)
    # -- Make sure to add the last token too
    if current_logprob_per_class and current_token_length:
        result_sequence += \
            current_token_length*[max(current_logprob_per_class, key=lambda cl: current_logprob_per_class[cl])]
    return result_sequence
