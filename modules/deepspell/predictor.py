# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import tensorflow as tf
import sys
import os
import json
import time

# ============================[ Local Imports ]==========================

from . import corpus
from . import grammar

# ====================[ Abstract Predictor Interface ]===================


class DSPredictor:

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, name_scope="unnamed", version=0, file_or_folder="", log_dir="", kwargs_to_update=None):
        """
        Instantiate or restore a Representation Model.
        :param name_scope: This will be included in the auto-generated model name,
         as well as the name for the variable scope of this model.
        :param version: This will be included in the auto-generated model name.
        :param file_or_folder: [optional] A file from which the model
         should be restored, or to which it should be saved.
        :param log_dir: The directory which will be served by tensor board,
         where tensor flow logs for this model should be written to.
        :param kwargs_to_update: [optional] A kwargs dictionary
         that should be updated with information stored in JSON-format
         under @file.
        """
        assert isinstance(file_or_folder, str)
        self.log_dir = log_dir
        self.file = file_or_folder
        self.tf_checkpoint_path = ""
        self.name_scope = name_scope
        self.version = version
        if os.path.isfile(file_or_folder) and isinstance(kwargs_to_update, dict):
            print("Loading model '{}'...".format(file_or_folder))
            assert os.path.splitext(file_or_folder)[1] == ".json"
            with open(file_or_folder, "r") as json_file:
                json_data = json.load(json_file)
                for key, value in json_data.items():
                    if key not in kwargs_to_update:
                        kwargs_to_update[key] = value
        elif file_or_folder:
            assert os.path.isdir(file_or_folder)
        self.batch_size = kwargs_to_update.pop("batch_size", 4096)
        self.iteration = kwargs_to_update.pop("iteration", 0)
        self.learning_rate = kwargs_to_update.pop("learning_rate", 0.003)
        self.learning_rate_decay = kwargs_to_update.pop("learning_rate_decay", 0.7)
        self.training_epochs = kwargs_to_update.pop("training_epochs", 10)
        self.training_history = kwargs_to_update.pop("training_history", [])
        self.num_lexical_features = kwargs_to_update.pop("num_lexical_features", 0)
        self.num_logical_features = kwargs_to_update.pop("num_logical_features", 0)

        # -- Create basic Tensor Flow nodes
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            self.tf_learning_rate = tf.placeholder(tf.float32)
            self.tf_lexical_logical_embeddings_per_timestep_per_batch = tf.placeholder(
                tf.float32,
                [None, None, self.num_logical_features + self.num_lexical_features])
            self.tf_lexical_logical_embeddings_per_timestep_per_batch_shape = tf.shape(
                self.tf_lexical_logical_embeddings_per_timestep_per_batch)
            self.tf_timesteps_per_batch = tf.placeholder(tf.int32, [None])
            self.tf_saver = None

    def store(self, file=None):
        """
        Store this model under a certain file path.
        :param file: The file path under which it should be stored.
        """
        params = self.info()
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

    def train(self, corpus, sample_grammar, train_test_split=None):
        """
        Train this model with documents from a given corpus.
        :param corpus: The corpus with documents to train from.
         Should be an instance of deepspell.corpus.Corpus.
        :param sample_grammar: The grammar that will be used to generate training examples.
        :param train_test_split: [optional] A 2-tuple that indicates the ratio
         for the train-test split of the corpus.
        """
        pass

    def name(self):
        """
        :return: This name will identify the log output from this model in Tensorboard.
        It will also serve as the name for all files (.json, .data-*, .index, .meta)
        that are associated with the model.
        """
        return "deepsp_{}-v{}_{}_lr{}_dec{}_bat{}".format(
            self.name_scope[:5],
            self.version,
            "+".join(self.training_history),
            str(self.learning_rate)[2:],
            str(int(self.learning_rate_decay * 100)),
            str(self.batch_size))

    def info(self):
        """
        This method will be called by save to determine values for variables that should
        be stored in the models json descriptor.
        :return: A dictionary that may be updated and returned by subclasses.
        """
        return {
            "learning_rate": self.learning_rate,
            "learning_rate_decay": self.learning_rate_decay,
            "training_epochs": self.training_epochs,
            "batch_size": self.batch_size,
            "training_history": self.training_history,
            "iteration": self.iteration,
            "num_lexical_features": self.num_lexical_features,
            "num_logical_features": self.num_logical_features
        }

    # ----------------------[ Private Methods ]----------------------

    def _train(self,
               train_op,
               summary_ops,
               training_corpus,
               sample_grammar,
               train_test_split,
               min_sample_length_before_truncation=-1):

        assert isinstance(training_corpus, corpus.DSCorpus)
        assert isinstance(sample_grammar, grammar.DSGrammar)
        assert self.num_lexical_features == training_corpus.num_lexical_features_per_character()
        assert self.num_logical_features == training_corpus.num_logical_features_per_character()

        with self.graph.as_default():

            self.training_history += [training_corpus.name]
            self.store()

            print("Training commencing for {}!".format(self.name()))
            print("------------------------------------------------------")
            current_learning_rate = self.learning_rate
            if os.path.isdir(self.log_dir):
                self.tf_summary_writer = tf.summary.FileWriter(os.path.join(self.log_dir, self.name()))
                self.tf_summary_writer.add_graph(self.graph)
            else:
                print("No valid log dir issued. Log will not be written!")

            ops = [train_op] + summary_ops
            epoch_leftover_documents = None
            epoch_count = 0
            start_time = time.time()

            # -- Commence training loop
            num_samples_done = 0
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
                    train_test_split,
                    min_sample_length_before_truncation)

                if print_iterator_size:
                    sys.stdout.write("Created new randomized collection from {} samples. Now training ...\n  ".format(
                        len(epoch_leftover_documents) + len(batch)))
                    print_iterator_size = False

                results = self.session.run(ops, feed_dict={
                    self.tf_lexical_logical_embeddings_per_timestep_per_batch: batch,
                    self.tf_timesteps_per_batch: lengths,
                    self.tf_learning_rate: current_learning_rate
                })

                if self.tf_summary_writer:
                    for summary_value in results[1:]:
                        self.tf_summary_writer.add_summary(summary_value, self.iteration)
                self.iteration += 1

                num_samples_done += len(batch)
                self._print_progress(num_samples_done, num_samples_done + len(epoch_leftover_documents))

                if not epoch_leftover_documents:
                    self.tf_saver.save(self.session, self.tf_checkpoint_path)
                    epoch_count += 1
                    current_learning_rate *= self.learning_rate_decay
                    print("\nModel saved in file:", self.tf_checkpoint_path)
                    self.store()
                    print("Epoch", epoch_count, "/", self.training_epochs, "complete after",
                          float(time.time() - start_time)/60.0, "minute(s).")
                    print("------------------------------------------------------")

        print("Optimization Finished!")

    def _finish_init(self):
        # Create saver only after the graph is initialized
        with self.graph.as_default():
            self.tf_saver = tf.train.Saver()
            self.tf_init_op = tf.global_variables_initializer()
            self.session.run(self.tf_init_op)
            print("Compute Graph Initialized.")
            print(" Trainable Variables:", tf.trainable_variables())
            if os.path.isfile(self.file):
                # -- Restore existing model checkpoint
                self.tf_checkpoint_path = os.path.splitext(self.file)[0]
                if os.path.isfile(self.tf_checkpoint_path+".index"):
                    print(" Restoring Tensor Flow Session from '{}'.".format(self.tf_checkpoint_path))
                    self.tf_saver.restore(self.session, self.tf_checkpoint_path)
            else:
                assert os.path.isdir(self.file)
                self.file = os.path.join(self.file, self.name() + ".json")
                self.tf_checkpoint_path = os.path.splitext(self.file)[0]

    # -*- coding: utf-8 -*-

    @staticmethod
    def _print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = '#' * filled_length + '-' * (bar_length - filled_length)

        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

        if iteration == total:
            sys.stdout.write('\n')

        sys.stdout.flush()
