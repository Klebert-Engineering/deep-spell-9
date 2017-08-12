import tensorflow as tf
import numpy as np
import os
import json
import time

# ============================[ Local Imports ]==========================

from . import abstract
from . import corpus

# =======================[ LSTM FTS Predictor Model ]=====================


class FtsLstmPredictor:

    # ------------------------[ Properties ]------------------------

    # -- Learning rate
    learning_rate = 0.007
    learning_rate_decay = 0.95

    # -- Number of training iterations
    training_epochs = 60

    # -- Number of examples per batch
    batch_size = 32

    # -- The size of the hidden states for the LSTM, per layer.
    #  Each LSTM layer uses 2x of this amount, one for C (state) and one for H (mem).
    state_size_per_layer = (256,)

    total_embedding_size = 0

    # ---------------------[ Interface Methods ]---------------------

    # -- Documentation in base Model class
    def __init__(self, embedding_model, file_or_folder, log_dir="", **kwargs):
        super().__init__(embedding_model, file_or_folder, kwargs)

        # -- Read params
        self.learning_rate = kwargs.pop("learning_rate", 0.007)
        self.learning_rate_decay = kwargs.pop("learning_rate_decay", 0.9)
        self.training_epochs = kwargs.pop("training_epochs", 60)
        self.batch_size = kwargs.pop("batch_size", 48)
        self.dropout_rate = kwargs.pop("dropout_rate", 0.2)
        self.state_size_per_layer = kwargs.pop("state_size_per_layer", (256, 256))
        self.embed_from_state_only = kwargs.pop("embed_from_state_only", True)

        # -- Create Tensor Flow compute graph nodes
        self.current_learning_rate = tf.placeholder(tf.float32)
        (self.values_per_batch_per_timestep,
         self.timesteps_per_batch,
         self.input_shape,
         self.enc_cell,
         self.enc_final_state) = self._encoder()
        self.train_op = self._optimizer()
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.init_op = tf.global_variables_initializer()
        if os.path.isdir(log_dir):
            self.summary_writer = tf.summary.FileWriter(os.path.join(log_dir, self.name()))
            self.summary_writer.add_graph(tf.get_default_graph())
        else:
            print("No valid log dir issued. Log will not be written!")
            self.summary_writer = None
        self.total_embedding_size = sum(self.state_size_per_layer) * (1 if self.embed_from_state_only else 2)
        self.session = tf.Session()
        print("Compute Graph Initialized.")
        print("   Trainable Variables:", tf.trainable_variables())

        # -- After params are loaded, auto-generate name for model file if necessary
        if os.path.isdir(file_or_folder):
            self.tf_checkpoint_path = os.path.join(file_or_folder, self.name())
            file_or_folder = os.path.join(file_or_folder, self.name()+".json")
            print("Creating '{}' ...".format(file_or_folder))
            self.file = file_or_folder
            self.store()
        else:
            # -- Otherwise restore existing model checkpoint
            self.tf_checkpoint_path = os.path.join(os.path.dirname(file_or_folder), self.name())
            if os.path.isfile(self.tf_checkpoint_path+".index"):
                print("Restoring Tensor Flow Session from '{}'.".format(self.tf_checkpoint_path))
                self.saver.restore(self.session, self.tf_checkpoint_path)

    # -- Documentation in base Model class
    def store(self, file=None):
        params = {
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
            "training_epochs": self.training_epochs,
            "batch_size": self.batch_size,
            "dropout_rate": self.dropout_rate,
            "state_size_per_layer": self.state_size_per_layer,
            "embed_from_state_only": self.embed_from_state_only,
        }
        if not file:
            file = self.file
        json.dump(params, open(file, 'w'), indent=4, sort_keys=True)

    # -- Documentation in base Model class
    def train(self, training_corpus, train_test_split=None):
        assert isinstance(training_corpus, corpus.Corpus)
        print("Training commencing for {}!".format(self.name()))
        current_learning_rate = self.learning_rate

        with self.session as sess:
            # -- Run value initialization op
            self.session.run(self.init_op)

            # -- Commence training loop
            epoch_leftover_documents = None
            epoch_count = 0
            iteration = 0
            start_time = time.time()
            while epoch_count < self.training_epochs:
                batch, lengths, epoch_leftover_documents = training_corpus.get_batch_and_lengths(
                    self.batch_size,
                    self.embedding_model,
                    epoch_leftover_documents,
                    train_test_split)
                _, summary = sess.run([self.train_op, self.summary_op], feed_dict={
                    self.values_per_batch_per_timestep: batch,
                    self.timesteps_per_batch: lengths,
                    self.current_learning_rate: current_learning_rate})
                if self.summary_writer:
                    self.summary_writer.add_summary(summary, iteration)
                iteration += 1

                # -- Store checkpoint per epoch.
                if not epoch_leftover_documents:
                    epoch_count += 1
                    current_learning_rate *= self.learning_rate_decay
                    print("Epoch", epoch_count, "/", self.training_epochs, "complete after",
                          float(time.time() - start_time)/60.0, "minute(s).")
                    self.saver.save(sess, self.tf_checkpoint_path)
                    print("  New learning rate:", current_learning_rate)
                    print("  Model saved in file:", self.tf_checkpoint_path)
        print("Optimization Finished!")

    # -- Documentation in base Model class
    def vector_size(self):
        return self.total_embedding_size

    def name(self):
        """
        :return: This name will identify the log output from this model in Tensorboard.
        It will also serve as the name for all files (.json, .ckpt.*) associated with the model.
        """
        return "LSTM-AE-LR{}DE{}-BAT{}-EMB{}-DROP{}".format(
            str(self.learning_rate)[2:],
            str(self.learning_rate_decay)[2:],
            str(self.batch_size),
            "".join(
                str(n) + ("C" if self.embed_from_state_only else "HC")
                for n in self.state_size_per_layer),
            str(int(self.dropout_rate * 100)))

    # ----------------------[ Private Methods ]----------------------

    def _encoder(self):
        # -- Input placeholders: time-first batch of training sequences and their lengths
        values_per_batch_per_timestep = tf.placeholder(
            tf.float32,
            [None, None, self.embedding_model.vector_size]
        )
        input_shape = tf.shape(values_per_batch_per_timestep)
        timesteps_per_batch = tf.placeholder(tf.int32, [None])

        # -- LSTM cell for encoding
        enc_cell = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
            self.state_size_per_layer
        ])
        enc_initial_state = enc_cell.zero_state(input_shape[1], tf.float32)

        # -- Create a dynamically unrolled RNN to produce the embedded document vector
        _, enc_final_state = tf.nn.dynamic_rnn(
            cell=enc_cell,
            inputs=values_per_batch_per_timestep,
            sequence_length=timesteps_per_batch,
            initial_state=enc_initial_state,
            time_major=True)
        return values_per_batch_per_timestep, timesteps_per_batch, input_shape, enc_cell, enc_final_state

    def _stepwise_encoder(self):
        initial_outputs = tf.TensorArray(dtype=tf.float32, size=self.input_shape[0])
        initial_t = tf.constant(0, dtype=tf.int32)
        initial_state = self.enc_cell.zero_state(self.input_shape[1], tf.float32)

        def should_continue(t, *args):
            return t < self.input_shape[0]

        def iteration(t, state, outputs):
            current_timestep_embeddings = tf.reshape(
                tf.gather(self.values_per_batch_per_timestep, t),
                (self.input_shape[1], self.embedding_model.vector_size)
            )
            with tf.variable_scope("rnn", reuse=True):
                _, state = self.enc_cell(
                    inputs=current_timestep_embeddings,
                    state=state)
            outputs = outputs.write(t, state)
            return t + 1, state, outputs

        _, _, final_outputs = tf.while_loop(
            should_continue, iteration,
            [initial_t, initial_state, initial_outputs]
        )
        return final_outputs.stack()

    def _optimizer(self):
        # -- Obtain global training step
        global_step = tf.contrib.framework.get_global_step()

        # -- Calculate the average log perplexity
        loss = tf.losses.mean_pairwise_squared_error(self.values_per_batch_per_timestep, self.decoder_outputs.rnn_output)

        # -- Define training op
        optimizer = tf.train.RMSPropOptimizer(self.current_learning_rate)
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=None,
            optimizer=optimizer)
        return train_op
