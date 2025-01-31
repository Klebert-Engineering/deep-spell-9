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
            for cell in tf_encoder_forward_cells:
                forward_outputs, forward_state_h, forward_state_c = cell(forward_outputs)
            
            # Backward pass through stacked LSTM
            for cell in tf_encoder_backward_cells:
                backward_outputs, backward_state_h, backward_state_c = cell(backward_outputs)
            backward_outputs = tf.reverse(backward_outputs, axis=[1])
            
            # Combine forward and backward outputs
            combined_outputs = tf.concat([forward_outputs, backward_outputs, tf_corrupt_encoder_input], axis=2)
            
            # Final processing through combine layers
            for cell in tf_encoder_combine_cells:
                combined_outputs, combine_state_h, combine_state_c = cell(combined_outputs)
            
            # Collect final states
            final_states = [combine_state_h, combine_state_c]
            tf_final_encoder_states_per_batch = tf.concat(final_states, axis=1)

        return tf_corrupt_encoder_input, tf_final_encoder_states_per_batch

    def _encoder_to_latent(self):
        """
        :return: tf_latent_mean, tf_latent_variance, tf_latent_vec
        """
        with tf.variable_scope('encoder_to_latent'):
            kl_rate = tf.placeholder(tf.float32, shape=[])
            concat_state_size = 2*sum(self.encoder_combine_state_size_per_layer)
            w = tf.get_variable("w", [concat_state_size, 2 * self.embedding_size], dtype=tf.float32)
            b = tf.get_variable("b", [2 * self.embedding_size], dtype=tf.float32)
            mean_logvar = self._prelu(tf.matmul(self.tf_encoder_final_state_per_batch, w) + b)
            means, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
            noise = tf.random_normal(tf.shape(means)) * kl_rate
            sampled_random_vectors = means + tf.exp(0.5 * logvar) * noise
            kl_loss = tf.reshape(
                tf.reduce_mean(-0.5 * (logvar - tf.square(means) - tf.exp(logvar) + 1.0),),
                shape=[])
            # if kl_min:
            #     kl_loss = tf.reduce_sum(tf.maximum(kl_ave, kl_min))
            kl_loss *= kl_rate
            kl_loss_summary = tf.summary.scalar("kl_loss", kl_loss)
        return sampled_random_vectors, means, kl_loss, kl_loss_summary, kl_rate
