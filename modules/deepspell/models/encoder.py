# (C) 2018-present Klebert Engineering

# ===============================[ Imports ]=============================

import base64
import codecs
import os
import pickle

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("WARNING: SciPy not installed!")
    cKDTree = None
    pass

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# ============================[ Local Imports ]==========================

from deepspell.models import modelbase
from deepspell import grammar


# =======================[ LSTM Extrapolator Model ]=====================

class DSVariationalLstmAutoEncoder(modelbase.DSModelBase):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", args_to_update=None, **kwargs):
        """Documentation in base Model class"""

        if not args_to_update:
            args_to_update = dict()
        args_to_update.update(kwargs)

        super().__init__(
            name_scope="spelling-encoder",
            version=2,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            args_to_update=args_to_update)

        # -- Read params
        self.encoder_fw_state_size_per_layer = args_to_update.pop("encoder_fw_state_size_per_layer", [128, 128])
        self.encoder_bw_state_size_per_layer = args_to_update.pop("encoder_bw_state_size_per_layer", [128, 128])
        self.encoder_combine_state_size_per_layer = args_to_update.pop("encoder_combine_state_size_per_layer", [128, 128])
        self.embedding_size = args_to_update.pop("embedding_size", 8)

        # -- Create Tensor Flow compute graph nodes
        with self.graph.as_default():
            (self.tf_corrupt_encoder_input,
             self.tf_encoder_final_state_per_batch) = self._encoder()
            (self.tf_latent_random_vectors,
             self.tf_latent_means,
             self.tf_kl_loss,
             self.tf_kl_loss_summary,
             self.tf_kl_rate) = self._encoder_to_latent()
        self._finish_init_base()

    def encode(self, string):
        with self.graph.as_default():
            embeddings = self.session.run(self.tf_latent_means, feed_dict={
                self.tf_corrupt_encoder_input: [
                    self.featureset.embed_characters(string)[:, :self.featureset.num_lexical_features()]],
                self.tf_timesteps_per_batch: [len(string)+1]
            })
            return embeddings[0]

    def encode_corpus(self, corpus_file_to_encode, batch_size=16384):
        if not cKDTree:
            print("WARNING: SciPy not installed!")
            return
        print("Encoding tokens from '{}' ...".format(corpus_file_to_encode))
        result_tokens = []
        known_tokens = set()
        token_embeddings = np.empty(shape=(0, self.embedding_size), dtype=np.float32)
        with codecs.open(corpus_file_to_encode) as corpus_file:
            total = sum(1 for _ in corpus_file)
        done = 0
        with codecs.open(corpus_file_to_encode, "r") as corpus_file:
            batch_tokens = []
            max_token_length = 0
            for line in corpus_file:
                parts = line.split("\t")
                if len(parts) < 6:
                    continue
                token = parts[2].lower()
                if token not in known_tokens:
                    known_tokens.add(token)
                    batch_tokens.append(grammar.DSToken(0, 0, None, token))
                    max_token_length = max(len(token) + 1, max_token_length)
                    result_tokens.append(token)
                else:
                    done += 1
                if len(batch_tokens) >= batch_size or done + len(batch_tokens) >= total:
                    batch_embedding_sequences, batch_lengths, _, _ = zip(*(
                        self.featureset.embed_token_sequence(
                            [token_object],
                            max_token_length+1,
                            embed_with_class=False)
                        for token_object in batch_tokens))
                    with self.graph.as_default():
                        encoder_state, embeddings = self.session.run([
                                self.tf_encoder_final_state_per_batch,
                                self.tf_latent_means],
                            feed_dict={
                                self.tf_corrupt_encoder_input: batch_embedding_sequences,
                                self.tf_timesteps_per_batch: batch_lengths
                            })
                    assert len(embeddings) == len(batch_tokens)
                    token_embeddings = np.concatenate((token_embeddings, embeddings), axis=0)
                    done += len(batch_tokens)
                    batch_tokens = []
                    max_token_length = 0
                    super()._print_progress(done, total)
        print("  ... done.")
        print("Building kd-tree ...")
        result_kdtree = cKDTree(token_embeddings)
        print("  ... done.")
        return result_tokens, result_kdtree

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
            tf_corrupt_encoder_input = tf.placeholder(
                tf.float32,
                [None, None, self.num_lexical_features])

            # Create backward LSTM layers
            tf_encoder_backward_cells = [
                tf.keras.layers.LSTM(hidden_state_size, return_sequences=True, return_state=True)
                for hidden_state_size in self.encoder_bw_state_size_per_layer
            ]
            
            # Create forward LSTM layers
            tf_encoder_forward_cells = [
                tf.keras.layers.LSTM(hidden_state_size, return_sequences=True, return_state=True)
                for hidden_state_size in self.encoder_fw_state_size_per_layer
            ]
            
            # Create combine LSTM layers
            tf_encoder_combine_cells = [
                tf.keras.layers.LSTM(hidden_state_size, return_sequences=True, return_state=True)
                for hidden_state_size in self.encoder_combine_state_size_per_layer
            ]

            # Process forward and backward passes
            forward_outputs = tf_corrupt_encoder_input
            backward_outputs = tf.reverse(tf_corrupt_encoder_input, axis=[1])
            
            # Forward pass through stacked LSTM
            forward_states = []
            for cell in tf_encoder_forward_cells:
                forward_outputs, forward_state_h, forward_state_c = cell(forward_outputs)
                forward_states.extend([forward_state_h, forward_state_c])
            
            # Backward pass through stacked LSTM
            backward_states = []
            for cell in tf_encoder_backward_cells:
                backward_outputs, backward_state_h, backward_state_c = cell(backward_outputs)
                backward_states.extend([backward_state_h, backward_state_c])
            backward_outputs = tf.reverse(backward_outputs, axis=[1])
            
            # Combine forward and backward outputs
            combined_outputs = tf.concat([forward_outputs, backward_outputs, tf_corrupt_encoder_input], axis=2)
            
            # Final processing through combine layers
            combine_states = []
            for cell in tf_encoder_combine_cells:
                combined_outputs, combine_state_h, combine_state_c = cell(combined_outputs)
                combine_states.extend([combine_state_h, combine_state_c])
            
            # Collect all states and concatenate
            all_states = forward_states + backward_states + combine_states
            tf_final_encoder_states_per_batch = tf.concat(all_states, axis=1)

        return tf_corrupt_encoder_input, tf_final_encoder_states_per_batch

    def _encoder_to_latent(self):
        """
        :return: tf_latent_mean, tf_latent_variance, tf_latent_vec
        """
        with tf.compat.v1.variable_scope('encoder_to_latent'):
            kl_rate = tf.placeholder(tf.float32, shape=[])
            concat_state_size = tf.shape(self.tf_encoder_final_state_per_batch)[1]
            w = tf.compat.v1.get_variable("w", [1536, 2 * self.embedding_size], dtype=tf.float32)
            b = tf.compat.v1.get_variable("b", [2 * self.embedding_size], dtype=tf.float32)
            mean_logvar = self._prelu(tf.matmul(self.tf_encoder_final_state_per_batch, w) + b)
            means, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
            noise = tf.random_normal(tf.shape(means)) * kl_rate
            sampled_random_vectors = means + tf.exp(0.5 * logvar) * noise
            kl_loss = tf.reshape(
                tf.reduce_mean(-0.5 * (logvar - tf.square(means) - tf.exp(logvar) + 1.0),),
                shape=[])
            kl_loss *= kl_rate
            kl_loss_summary = tf.summary.scalar("kl_loss", kl_loss)
        return sampled_random_vectors, means, kl_loss, kl_loss_summary, kl_rate

    def _latent_to_stepwise_decoder(self):
        """
        :return: tf_eol_char_id, tf_unk_char_id, tf_start_char_id, tf_correct_decoder_output, tf_stepwise_decoder_output
        """
        with tf.compat.v1.variable_scope('stepwise_decoder'):
            tf_eol_char_id = tf.placeholder(tf.int32, shape=[])
            tf_unk_char_id = tf.placeholder(tf.int32, shape=[])
            tf_start_char_id = tf.placeholder(tf.int32, shape=[])
            tf_correct_decoder_output = tf.placeholder(
                dtype=tf.float32,
                shape=(None, None, self.featureset.num_lexical_features()))
            tf_batch_size = tf.shape(tf_correct_decoder_output)[0]

            # Create decoder LSTM layers
            tf_decoder_cells = [
                tf.keras.layers.LSTM(hidden_state_size, return_sequences=True, return_state=True)
                for hidden_state_size in self.decoder_state_size_per_layer
            ]
            
            # Create output projection layer
            tf_output_projection = tf.keras.layers.Dense(self.featureset.num_lexical_features())

            if self.latent_space_as_decoder_state:
                with tf.compat.v1.variable_scope('latent_to_decoder'):
                    concat_state_size = sum(n*2 for n in self.decoder_state_size_per_layer)
                    tf_w = tf.compat.v1.get_variable("w", [self.embedding_size, concat_state_size], dtype=tf.float32)
                    tf_b = tf.compat.v1.get_variable("b", [concat_state_size], dtype=tf.float32)
                    tf_decoder_initial_state = self._prelu(tf.matmul(self.tf_latent_random_vectors, tf_w) + tf_b)
                    
                    # Split initial state for each layer
                    pos_in_state = 0
                    initial_states = []
                    for state_size in self.decoder_state_size_per_layer:
                        h = tf_decoder_initial_state[:, pos_in_state:pos_in_state+state_size]
                        c = tf_decoder_initial_state[:, pos_in_state+state_size:pos_in_state+2*state_size]
                        initial_states.append([h, c])
                        pos_in_state += state_size*2  # *2 for state+mem
            else:
                initial_states = None

            tf_max_decoder_steps = tf.shape(tf_correct_decoder_output)[1]
            tf_stepwise_decoder_output = tf.TensorArray(tf.float32, size=1, dynamic_size=True)
            tf_prev_output = tf.reshape(tf.tile(
                tf.one_hot(tf_start_char_id, depth=self.featureset.num_lexical_features()),
                [tf_batch_size]), shape=(-1, self.featureset.num_lexical_features()))

            def should_continue(t, *_):
                return t < tf_max_decoder_steps

            def iteration(t, prev_output, states, stepwise_decoder_output):
                input_keep_prob = tf.random_uniform([], .0, 1.)
                prev_output = tf.cond(
                    tf.logical_and(input_keep_prob > self.decoder_input_keep_prob, t > 0),
                    lambda: tf.reshape(tf.tile(
                        tf.one_hot(tf_unk_char_id, depth=self.featureset.num_lexical_features()),
                        [tf_batch_size]), shape=(-1, self.featureset.num_lexical_features())),
                    lambda: prev_output)
                
                # Concatenate with latent vector
                decoder_input = tf.concat([prev_output, self.tf_latent_random_vectors], axis=1)
                
                # Process through LSTM layers
                current_input = decoder_input
                new_states = []
                for i, (cell, state) in enumerate(zip(tf_decoder_cells, states)):
                    current_input = tf.expand_dims(current_input, axis=1)  # Add time dimension
                    outputs, state_h, state_c = cell(current_input, initial_state=state)
                    current_input = tf.squeeze(outputs, axis=1)  # Remove time dimension
                    new_states.append([state_h, state_c])
                
                # Project output
                current_output = tf_output_projection(current_input)
                
                # Write to output array
                stepwise_decoder_output = stepwise_decoder_output.write(t, current_output)
                
                # Use teacher forcing
                prev_output = tf_correct_decoder_output[:, t, :]
                
                return t+1, prev_output, new_states, stepwise_decoder_output

            # Initialize states if not provided
            if initial_states is None:
                initial_states = [[None, None] for _ in tf_decoder_cells]

            _, _, _, tf_stepwise_decoder_output = tf.while_loop(
                should_continue,
                iteration,
                [tf.constant(0), tf_prev_output, initial_states, tf_stepwise_decoder_output])

        tf_stepwise_decoder_output = tf.transpose(tf_stepwise_decoder_output.stack(), perm=(1, 0, 2))
        return tf_eol_char_id, tf_unk_char_id, tf_start_char_id, tf_correct_decoder_output, tf_stepwise_decoder_output
