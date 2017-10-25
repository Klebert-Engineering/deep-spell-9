# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import json
import os
import sys
import time

import tensorflow as tf

# ============================[ Local Imports ]==========================

from deepspell import corpus
from deepspell.models import modelbase
from deepspell_optimization import grammar


# ====================[ Abstract Predictor Interface ]===================

class DSModelOptimizerMixin(modelbase.DSModelBase):

    def __init__(self, args):
        """
        Utilize multiple inheritence with this class and a class derived from
        `modelbase.DSModelBase`. Therefore, modelbase.DSModelBase.__init__()
        is *explicitely not* called here, but should be called by the inherited
        <Model>, which DSModelOptimizerMixin is mixed-in with into the <ModelOptimizer>:

            ```
            (deepspell)
                                 /-----------------------\
                                 | DSModelBase           |
                                 | + graph: tf.Graph     |
                                 | + finish_init_base()  |
                                 \----A------------A-----/
                                      '            | __init__
                                      '       /---------\
                                      '       | <Model> |
                                      '       \----A----/
                                      '            |
            - - - - - - - - - - - - - ' - - - - - -|- - - - - - - - -
            (deepspell_optimization)  '            |
                                      '            |
                    /-----------------'---------\  |
                    | DSModelOptimizerMixin     |  |
                    | + finish_init_base() {}   |  |
                    | + finish_init_optimizer() |  |
                    | + _train(...)             |  |
                    \-----------------A---------/  |
                                      | __init__#1 | __init__#0
                                   /------------------\
                                   | <ModelOptimizer> |
                                   \------------------/
            ```

        Note:
            - It is important, that `DSModelOptimizerMixin` is the *first inherited class*.
              This is, because `DSModelOptimizerMixin` overwrites finish_init_base() from
              `DSModelBase` with an empty impl. Instead, it provides finish_init_optimizer().
            - It is important, that `DSModelBase` is initialized before `DSModelOptimizerMixin`,
              because DSModelBase creates `self.graph`, which DSModelOptimizerMixin relies on.
              Therefore, *do not use super().__init__(...)*, but explicit `DSModelBase.__init__(...)` and
              `DSModelOptimizerMixin.__init__(...)` calls.
        """
        self.batch_size = args.pop("batch_size", 4096)
        self.iteration = args.pop("iteration", 0)
        self.learning_rate = args.pop("learning_rate", 0.003)
        self.learning_rate_decay = args.pop("learning_rate_decay", 0.7)
        self.training_epochs = args.pop("training_epochs", 10)
        self.training_history = args.pop("training_history", [])
        self.tf_summary_writer = None
        with self.graph.as_default():
            self.tf_learning_rate = tf.placeholder(tf.float32)

    def store(self, file=None):
        """
        Store this model under a certain file path.
        :param file: The file path under which it should be stored.
        """
        params = self.info()
        if not file:
            folder = os.path.dirname(self.file)
            assert os.path.isdir(folder)
            file = os.path.join(folder, self.generate_name() + ".json")
        assert os.path.splitext(file)[1] == ".json"
        self.file = file
        self.tf_checkpoint_path = os.path.splitext(file)[0]
        if os.path.isfile(file):
            print("Overwriting '{}' ...".format(self.tf_checkpoint_path))
        else:
            print("Creating '{}' ...".format(self.tf_checkpoint_path))
        with open(file, 'w') as open_file:
            json.dump(params, open_file, indent=2, sort_keys=True)

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
            "features": self.featureset.as_dict()
        }

    def generate_name(self):
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

    # ----------------------[ Private Methods ]----------------------

    def _finish_init_base(self):
        pass

    def _finish_init_optimizer(self):
        if os.path.isdir(self.file):
            self.file = os.path.join(self.file, self.generate_name() + ".json")
            self.tf_checkpoint_path = os.path.splitext(self.file)[0]
        modelbase.DSModelBase._finish_init_base(self)

    def _train(self,
               train_ops,
               summary_ops,
               training_corpus,
               sample_grammar,
               train_test_split,
               min_sample_length_before_truncation=-1,
               feed_fn=None,
               run_callback_fn=None):

        assert isinstance(training_corpus, corpus.DSCorpus)
        assert isinstance(sample_grammar, grammar.DSGrammar)
        self.featureset.adapt_logical_features(training_corpus.featureset)

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

            if not isinstance(train_ops, list):
                train_ops = [train_ops]
            ops = train_ops + summary_ops
            epoch_leftover_documents = None
            epoch_count = 0
            start_time = time.time()

            # -- Commence training loop
            num_samples_done = 0
            print_iterator_size = False

            def default_feed_fn(epoch_it, learning_rate):
                batch, lengths, epoch_it, *_ = training_corpus.next_batches_and_lengths(
                    self.batch_size,
                    sample_grammar,
                    epoch_it,
                    train_test_split,
                    min_sample_length_before_truncation)
                return {
                    self.tf_lexical_logical_embeddings_per_timestep_per_batch: batch,
                    self.tf_timesteps_per_batch: lengths,
                    self.tf_learning_rate: learning_rate
                }, epoch_it
            if not feed_fn:
                feed_fn = default_feed_fn

            while epoch_count < self.training_epochs:

                if not epoch_leftover_documents:
                    print("New epoch at learning rate {}.".format(current_learning_rate))
                    num_samples_done = 0
                    print_iterator_size = True

                feed, epoch_leftover_documents = feed_fn(epoch_leftover_documents, current_learning_rate)

                if print_iterator_size:
                    sys.stdout.write("Created new randomized collection from {} samples. Now training ...\n  ".format(
                        len(epoch_leftover_documents) + self.batch_size))
                    print_iterator_size = False

                results = self.session.run(ops, feed_dict=feed)
                if run_callback_fn:
                    run_callback_fn(results)

                if self.tf_summary_writer:
                    for summary_value in results[len(train_ops):]:
                        self.tf_summary_writer.add_summary(summary_value, self.iteration)
                self.iteration += 1

                # -- Note, that this will overshoot for the last batch. use with care.
                num_samples_done += self.batch_size
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
