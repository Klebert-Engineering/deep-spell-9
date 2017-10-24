# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import base64
import codecs
import os

import numpy as np
import tensorflow as tf

# ============================[ Local Imports ]==========================

from deepspell import corpus
from deepspell.models import modelbase


# =======================[ LSTM Extrapolator Model ]=====================

class DSVariationalLstmAutoEncoder(modelbase.DSModelBase):

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class"""
        super().__init__(
            name_scope="spelling-encoder",
            version=1,
            file_or_folder=file_or_folder,
            log_dir=log_dir,
            kwargs_to_update=kwargs)

        # -- Read params
        self.encoder_fw_state_size_per_layer = kwargs.pop("encoder_fw_state_size_per_layer", [128, 128])
        self.encoder_bw_state_size_per_layer = kwargs.pop("encoder_bw_state_size_per_layer", [128, 128])
        self.embedding_size = kwargs.pop("embedding_size", 8)

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

    def encode_corpus(self, corpus_to_encode, batch_size, output_path):
        assert isinstance(corpus_to_encode, corpus.DSCorpus)
        result_vectors = []
        tokens = [
            token
            for class_id, tokens in corpus_to_encode.data.items()
            for token in tokens]
        total = len(tokens)
        done = 0

        # -- Find free output path
        file_path = os.path.join(output_path, corpus_to_encode.generate_name + ".{}.vectors.bin")
        i = 0
        while os.path.exists(file_path.format(i)):
            i += 1
        file_path = file_path.format(i)
        print("Dumping encoded tokens to file '{}'".format(file_path))
        with codecs.open(file_path, "wb") as dump_file:
            print("")
            while tokens:
                batch_tokens = tokens[:batch_size]
                tokens = tokens[batch_size:]
                done += len(batch_tokens)
                super()._print_progress(done, total)
                # -- len(token.string)+1 to account for EOL
                max_token_length = max(len(token.string)+1 for token in batch_tokens)
                batch_embedding_sequences, batch_lengths, _, _ = zip(*(
                    self.featureset.embed_tokens(
                        [token],
                        max_token_length,
                        -1,
                        corruption_grammar=None,
                        embed_with_class=False)
                    for token in batch_tokens))
                with self.graph.as_default():
                    embeddings = self.session.run(self.tf_latent_means, feed_dict={
                        self.tf_corrupt_encoder_input: batch_embedding_sequences,
                        self.tf_timesteps_per_batch: batch_lengths
                    })
                assert len(embeddings) == len(batch_tokens)
                for embedding, token in zip(embeddings, batch_tokens):
                    # print(token.string, ":", embedding)
                    dump_file.write(codecs.encode(token.string)+b"\t")
                    dump_file.write(base64.b64encode(embedding.tostring())+b"\n")
            print("")

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

            # -- Backward pass
            with tf.variable_scope("backward"):
                tf_encoder_backward_cell = tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                    self.encoder_bw_state_size_per_layer])
                tf_backward_embeddings_per_timestep_per_batch, tf_final_bw_state_tuple_stack = tf.nn.dynamic_rnn(
                    cell=tf_encoder_backward_cell,
                    inputs=tf.reverse(tf_corrupt_encoder_input, axis=[1]),
                    initial_state=tf_encoder_backward_cell.zero_state(
                        tf.shape(tf_corrupt_encoder_input)[0],
                        tf.float32),
                    time_major=False)

            with tf.variable_scope("forward"):
                # -- Forward pass
                tf_discriminator_forward_cell = tf.contrib.rnn.MultiRNNCell([
                        tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                        self.encoder_fw_state_size_per_layer])
                # -- Create a dynamically unrolled RNN to produce the character category discrimination
                tf_logical_predictions_per_timestep_per_batch, tf_final_fw_state_tuple_stack = tf.nn.dynamic_rnn(
                    cell=tf_discriminator_forward_cell,
                    inputs=tf.concat([
                        tf_corrupt_encoder_input,
                        tf.reverse(tf_backward_embeddings_per_timestep_per_batch, axis=[1])], axis=2),
                    sequence_length=self.tf_timesteps_per_batch,
                    initial_state=tf_discriminator_forward_cell.zero_state(
                        tf.shape(tf_corrupt_encoder_input)[0],
                        tf.float32),
                    time_major=False)

        concat_state_size = sum(
            n * 2 for n in self.encoder_fw_state_size_per_layer + self.encoder_bw_state_size_per_layer)
        tf_final_encoder_states_per_batch = tf.reshape(tf.concat([
            state
            for state_tuple_stack in (tf_final_fw_state_tuple_stack, tf_final_bw_state_tuple_stack)
            for state_tuple in state_tuple_stack
            for state in state_tuple], axis=1), shape=(-1, concat_state_size))

        return tf_corrupt_encoder_input, tf_final_encoder_states_per_batch

    def _encoder_to_latent(self):
        """
        :return: tf_latent_mean, tf_latent_variance, tf_latent_vec
        """
        with tf.variable_scope('encoder_to_latent'):
            kl_rate = tf.placeholder(tf.float32, shape=[])
            concat_state_size = sum(
                n*2 for n in self.encoder_fw_state_size_per_layer+self.encoder_bw_state_size_per_layer)
            w = tf.get_variable("w", [concat_state_size, 2 * self.embedding_size], dtype=tf.float32)
            b = tf.get_variable("b", [2 * self.embedding_size], dtype=tf.float32)
            mean_logvar = self._prelu(tf.matmul(self.tf_encoder_final_state_per_batch, w) + b)
            means, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
            noise = tf.random_normal(tf.shape(means))
            sampled_random_vectors = means + tf.exp(0.5 * logvar) * noise
            kl_loss = tf.reshape(
                tf.reduce_mean(-0.5 * (logvar - tf.square(means) - tf.exp(logvar) + 1.0),),
                shape=[])
            # if kl_min:
            #     kl_loss = tf.reduce_sum(tf.maximum(kl_ave, kl_min))
            kl_loss *= kl_rate
            kl_loss_summary = tf.summary.scalar("kl_loss", kl_loss)
        return sampled_random_vectors, means, kl_loss, kl_loss_summary, kl_rate
