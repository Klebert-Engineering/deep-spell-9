# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import tensorflow as tf
import sys
import os
import json
import time
import numpy as np

# ============================[ Local Imports ]==========================

from . import abstract
from . import corpus
from . import grammar


# =======================[ LSTM FTS Predictor Model ]=====================


class DSLstmPredictor(abstract.DSPredictor):

    VERSION = 1

    # ------------------------[ Properties ]------------------------

    # -- Directory to which Tensor Flow logs from this graph will be written
    log_dir = ""

    # -- Learning rate
    learning_rate = 0.007
    learning_rate_decay = 0.95

    # -- Number of training iterations
    training_epochs = 60

    # -- Number of examples per batch
    batch_size = 32

    # -- The size of the hidden states for the LSTM, per layer.
    #  Each LSTM layer uses 2x of this amount, one for C (state) and one for H (mem).
    extrapolator_state_size_per_layer = (256,)

    # -- This property indicates the corpora this model has been trained on
    #  in the past.
    training_history = []

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, file_or_folder, log_dir="", **kwargs):
        """Documentation in base Model class
        :param total_features_per_character: The exact number of features per character,
         which is required for the Graph construction.
        """
        super().__init__(file_or_folder, kwargs)
        self.log_dir = log_dir

        # -- Read params
        self.learning_rate = kwargs.pop("learning_rate", 0.007)
        self.learning_rate_decay = kwargs.pop("learning_rate_decay", 0.9)
        self.training_epochs = kwargs.pop("training_epochs", 60)
        self.batch_size = kwargs.pop("batch_size", 48)
        self.extrapolator_state_size_per_layer = kwargs.pop("extrapolator_state_size_per_layer", (256, 256))
        self.discriminator_forward_state_size_per_layer = kwargs.pop("discriminator_forward_state_size_per_layer", (256, 256))
        self.discriminator_backward_state_size_per_layer = kwargs.pop("discriminator_backward_state_size_per_layer", (256, 256))
        self.training_history = kwargs.pop("training_history", [])
        self.iteration = kwargs.pop("iteration", 0)
        self.num_lexical_features = kwargs.pop("num_lexical_features", 0)
        self.num_logical_features = kwargs.pop("num_logical_features", 0)

        # -- Create Tensor Flow compute graph nodes
        self.tf_learning_rate = tf.placeholder(tf.float32)
        (self.tf_lexical_logical_embeddings_per_timestep_per_batch,
         self.tf_timesteps_per_batch,
         self.tf_num_lexical_classes,
         self.tf_num_logical_classes,
         self.tf_lexical_logical_embeddings_per_timestep_per_batch_shape,
         self.tf_extrapolator_cell,
         self.tf_lexical_logical_predictions_per_timestep_per_batch,
         self.tf_extrapolator_final_state_and_mem_stack) = self._extrapolator()

        (self.tf_lexical_embeddings_per_timestep_per_batch,
         self.tf_logical_predictions_per_timestep_per_batch) = self._discriminator()

        (self.tf_maximum_stepwise_extrapolation_length,
         self.tf_stepwise_extrapolator_output) = self._stepwise_extrapolator()

        (self.tf_train_op,
         self.tf_logical_loss_summary,
         self.tf_lexical_loss_summary) = self._extrapolator_optimizer()

        self.tf_saver = tf.train.Saver()
        self.tf_init_op = tf.global_variables_initializer()
        self.tf_summary_writer = None  # The summary writer will be created when training starts
        self.session = tf.Session()
        self.session.run(self.tf_init_op)

        print("Compute Graph Initialized.")
        print(" Trainable Variables:", tf.trainable_variables())
        if os.path.isfile(file_or_folder):
            # -- Restore existing model checkpoint
            self.tf_checkpoint_path = os.path.splitext(file_or_folder)[0]
            if os.path.isfile(self.tf_checkpoint_path+".index"):
                print(" Restoring Tensor Flow Session from '{}'.".format(self.tf_checkpoint_path))
                self.tf_saver.restore(self.session, self.tf_checkpoint_path)
        else:
            assert os.path.isdir(file_or_folder)
            self.file = os.path.join(file_or_folder, self.name() + ".json")
            self.tf_checkpoint_path = os.path.splitext(self.file)[0]

    def store(self, file=""):
        """Documentation in base Model class"""
        assert isinstance(file, str)
        params = {
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
            "training_epochs": self.training_epochs,
            "batch_size": self.batch_size,
            "extrapolator_state_size_per_layer": self.extrapolator_state_size_per_layer,
            "discriminator_forward_state_size_per_layer": self.discriminator_forward_state_size_per_layer,
            "discriminator_backward_state_size_per_layer": self.discriminator_backward_state_size_per_layer,
            "training_history": self.training_history,
            "iteration": self.iteration,
            "num_lexical_features": self.num_lexical_features,
            "num_logical_features": self.num_logical_features
        }
        if not file:
            folder = os.path.dirname(self.file)
            assert os.path.isdir(folder)
            file = os.path.join(folder, self.name() + ".json")

        assert os.path.splitext(file)[1] == ".json"
        self.file = file
        self.tf_checkpoint_path = os.path.splitext(file)[0]
        if os.path.isfile(file):
            print("Overwriting '{}' ...".format(self.tf_checkpoint_path))
        else:
            print("Creating '{}' ...".format(self.tf_checkpoint_path))

        with open(file, 'w') as open_file:
            json.dump(params, open_file, indent=2, sort_keys=True)

    def train_extrapolator(self, training_corpus, sample_grammar, train_test_split=None):
        """Documentation in base Model class"""
        assert isinstance(training_corpus, corpus.DSCorpus)
        assert isinstance(sample_grammar, grammar.DSGrammar)
        assert self.num_lexical_features == training_corpus.total_num_lexical_features_per_character()
        assert self.num_logical_features == training_corpus.total_num_logical_features_per_character()
        self.training_history += [training_corpus.name]
        self.store()

        print("Extrapolator Training commencing for {}!".format(self.name()))
        print("------------------------------------------------------")
        current_learning_rate = self.learning_rate
        if os.path.isdir(self.log_dir):
            self.tf_summary_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.name()))
            self.tf_summary_writer.add_graph(tf.get_default_graph())
        else:
            print("No valid log dir issued. Log will not be written!")

        with self.session as sess:
            epoch_leftover_documents = None
            epoch_count = 0
            start_time = time.time()

            # -- Commence training loop
            num_samples_done = 0
            prev_percent_done = 0
            print_iterator_size = False

            while epoch_count < self.training_epochs:
                if not epoch_leftover_documents:
                    print("New epoch at learning rate {}.".format(current_learning_rate))
                    num_samples_done = 0
                    prev_percent_done = 0
                    print_iterator_size = True

                batch, lengths, epoch_leftover_documents = training_corpus.get_batch_and_lengths(
                    self.batch_size,
                    sample_grammar,
                    epoch_leftover_documents,
                    train_test_split)

                if print_iterator_size:
                    sys.stdout.write("Created new randomized collection from {} samples. Now training ...\n  ".format(
                        len(epoch_leftover_documents) + len(batch)))
                    print_iterator_size = False

                _, logical_loss_value, lexical_loss_value = sess.run(
                    [
                        self.tf_train_op,
                        self.tf_logical_loss_summary,
                        self.tf_lexical_loss_summary
                    ],
                    feed_dict={
                        self.tf_lexical_logical_embeddings_per_timestep_per_batch: batch,
                        self.tf_timesteps_per_batch: lengths,
                        self.tf_learning_rate: current_learning_rate,
                        self.tf_num_lexical_classes: self.num_lexical_features,
                        self.tf_num_logical_classes: self.num_logical_features
                    })

                if self.tf_summary_writer:
                    self.tf_summary_writer.add_summary(logical_loss_value, self.iteration)
                    self.tf_summary_writer.add_summary(lexical_loss_value, self.iteration)
                self.iteration += 1

                num_samples_done += len(batch)
                percent_done = int(float(num_samples_done) / float(num_samples_done + len(epoch_leftover_documents)) * 100)
                if percent_done > prev_percent_done:  # - 10
                    # prev_percent_done = int(percent_done / 10) * 10
                    prev_percent_done = percent_done
                    sys.stdout.write("{}% ({}/{}) .. ".format(
                        prev_percent_done,
                        num_samples_done,
                        num_samples_done + len(epoch_leftover_documents)))
                    sys.stdout.flush()

                if not epoch_leftover_documents:
                    self.tf_saver.save(sess, self.tf_checkpoint_path)
                    epoch_count += 1
                    current_learning_rate *= self.learning_rate_decay
                    print("\nModel saved in file:", self.tf_checkpoint_path)
                    self.store()
                    print("Epoch", epoch_count, "/", self.training_epochs, "complete after",
                          float(time.time() - start_time)/60.0, "minute(s).")
                    print("------------------------------------------------------")

        print("Optimization Finished!")

    def extrapolate(self, completion_corpus, prefix_chars, prefix_classes, num_chars_to_predict):
        """Documentation in base Model class"""
        assert isinstance(completion_corpus, corpus.DSCorpus)
        assert len(prefix_chars) == len(prefix_classes)
        assert self.num_lexical_features == completion_corpus.total_num_lexical_features_per_character()
        assert self.num_logical_features == completion_corpus.total_num_logical_features_per_character()

        embedded_prefix = np.reshape(
            completion_corpus.embed_prefix(prefix_chars, prefix_classes),
            newshape=(1, len(prefix_chars), self.num_logical_features + self.num_lexical_features))

        stepwise_extrapolator_output = self.session.run(self.tf_stepwise_extrapolator_output, feed_dict={
            self.tf_lexical_logical_embeddings_per_timestep_per_batch: embedded_prefix,
            self.tf_timesteps_per_batch: np.asarray([len(prefix_chars)]),
            self.tf_num_lexical_classes: self.num_lexical_features,
            self.tf_num_logical_classes: self.num_logical_features,
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

    # -----------------------[ Public Methods ]----------------------

    def name(self):
        """
        :return: This name will identify the log output from this model in Tensorboard.
        It will also serve as the name for all files (.json, .data-*, .index, .meta)
        that are associated with the model.
        """
        return "deepspell_lstm_v{}_{}_lr{}_dec{}_bat{}".format(
            self.VERSION,
            "+".join(self.training_history),
            str(self.learning_rate)[2:],
            str(int(self.learning_rate_decay*100)),
            str(self.batch_size))

    # ----------------------[ Private Methods ]----------------------

    def _extrapolator(self):
        # -- Input placeholders: batch of training sequences and their lengths
        with tf.name_scope("extrapolator"):
            tf_lexical_logical_embeddings_per_timestep_per_batch = tf.placeholder(
                tf.float32,
                [None, None, self.num_logical_features + self.num_lexical_features])
            tf_num_lexical_classes = tf.placeholder(tf.int32)
            tf_num_logical_classes = tf.placeholder(tf.int32)
            tf_input_shape = tf.shape(tf_lexical_logical_embeddings_per_timestep_per_batch)
            tf_timesteps_per_batch = tf.placeholder(tf.int32, [None])

            # -- Ensure that input shape matches logical/lexical class split!
            tf_verify_logical_lexical_split_op = tf.Assert(
                tf.equal(tf_num_lexical_classes+tf_num_logical_classes, tf_input_shape[2]),
                [tf_num_lexical_classes, tf_num_logical_classes])

            with tf.control_dependencies([tf_verify_logical_lexical_split_op]):

                # -- LSTM cell for prediction
                tf_extrapolator_cell = tf.contrib.rnn.OutputProjectionWrapper(
                    tf.contrib.rnn.MultiRNNCell([
                        tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                        self.extrapolator_state_size_per_layer
                    ]),
                    self.num_logical_features + self.num_lexical_features
                )
                extrapolator_initial_state = tf_extrapolator_cell.zero_state(tf_input_shape[0], tf.float32)

                # -- Create a dynamically unrolled RNN to produce the embedded document vector
                tf_lexical_logical_predictions_per_timestep_per_batch, tf_final_state = tf.nn.dynamic_rnn(
                    cell=tf_extrapolator_cell,
                    inputs=tf_lexical_logical_embeddings_per_timestep_per_batch,
                    sequence_length=tf_timesteps_per_batch,
                    initial_state=extrapolator_initial_state,
                    time_major=False)

        return (
            tf_lexical_logical_embeddings_per_timestep_per_batch,
            tf_timesteps_per_batch,
            tf_num_lexical_classes,
            tf_num_logical_classes,
            tf_input_shape,
            tf_extrapolator_cell,
            tf_lexical_logical_predictions_per_timestep_per_batch,
            tf_final_state)

    def _discriminator(self):
        with tf.name_scope("discriminator"):
            # -- Input placeholders: batch of training sequences and their lengths
            tf_lexical_embeddings_per_timestep_per_batch = tf.placeholder(
                tf.float32,
                [None, None, self.num_lexical_features])
            tf_input_shape = tf.shape(tf_lexical_embeddings_per_timestep_per_batch)

            # -- Backward pass

            # -- Forward pass
            tf_discriminator_forward_cell = tf.contrib.rnn.OutputProjectionWrapper(
                tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(hidden_state_size) for hidden_state_size in
                    self.discriminator_forward_state_size_per_layer
                ]),
                self.num_logical_features
            )
            # -- Create a dynamically unrolled RNN to produce the character category discrimination
            tf_logical_predictions_per_timestep_per_batch, _ = tf.nn.dynamic_rnn(
                cell=tf_discriminator_forward_cell,
                inputs=tf_lexical_embeddings_per_timestep_per_batch,
                sequence_length=self.tf_timesteps_per_batch,
                initial_state=tf_discriminator_forward_cell.zero_state(tf_input_shape[0], tf.float32),
                time_major=False)

        return (
            tf_lexical_embeddings_per_timestep_per_batch,
            tf_logical_predictions_per_timestep_per_batch)

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
                            tf.argmax(prev_output[:, :self.tf_num_lexical_classes], axis=1),
                            depth=self.tf_num_lexical_classes),
                        tf.one_hot(
                            tf.argmax(prev_output[:, -self.tf_num_logical_classes:], axis=1),
                            depth=self.tf_num_logical_classes),
                    ],
                    axis=1), shape=(1, self.num_lexical_features+self.num_logical_features))

                # -- Softmax and flatten prediction because only a single batch is actually predicted
                predictions_per_timestep = predictions_per_timestep.write(t, tf.reshape(tf.concat([
                    tf.nn.softmax(prev_output[:, :self.tf_num_lexical_classes], dim=1),
                    tf.nn.softmax(prev_output[:, -self.tf_num_logical_classes:], dim=1)],
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

            # Time indices are sliced aas follows:
            #  For labels: First Input can be ignored
            #  For predictions: Last output (after EOD) can be ignored
            #
            # E.g.:
            #  Label = A B C D E A . 0 0
            #  Pred  = B C D E A . 0 0 0
            #  Slice First Input from Label, Last Output from Prediction:
            #  Label = B C D E A . 0 0
            #  Pred  = B C D E A . 0 0

            # -- Calculate the average cross entropy for the logical classes per timestep
            tf_logical_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                 labels=self.tf_lexical_logical_embeddings_per_timestep_per_batch[:, 1:, -self.tf_num_logical_classes:],
                 logits=self.tf_lexical_logical_predictions_per_timestep_per_batch[:, :-1, -self.tf_num_logical_classes:],
                 dim=2))

            # -- Calculate the average cross entropy for the lexical classes per timestep
            tf_lexical_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=self.tf_lexical_logical_embeddings_per_timestep_per_batch[:, 1:, :self.tf_num_lexical_classes],
                logits=self.tf_lexical_logical_predictions_per_timestep_per_batch[:, :-1, :self.tf_num_lexical_classes],
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
