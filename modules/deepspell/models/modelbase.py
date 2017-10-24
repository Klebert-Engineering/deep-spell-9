# (C) 2017 Klebert Engineering GmbH

# ===============================[ Imports ]=============================

import json
import os
import sys
import time

import tensorflow as tf

# ============================[ Local Imports ]==========================

from deepspell import featureset


# ====================[ Abstract Predictor Interface ]===================

class DSModelBase:

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
        featureset_params = kwargs_to_update.pop("features", None)
        if isinstance(featureset_params, dict):
            self.featureset = featureset.DSFeatureSet(**featureset_params)
        elif isinstance(featureset_params, featureset.DSFeatureSet):
            self.featureset = featureset_params
        else:
            self.featureset = featureset.DSFeatureSet()
        self.num_logical_features = self.featureset.num_logical_features()
        self.num_lexical_features = self.featureset.num_lexical_features()

        # -- Create basic Tensor Flow nodes
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session()
            self.tf_lexical_logical_embeddings_per_timestep_per_batch = tf.placeholder(
                tf.float32,
                [None, None, self.num_logical_features + self.num_lexical_features])
            self.tf_lexical_logical_embeddings_per_timestep_per_batch_shape = tf.shape(
                self.tf_lexical_logical_embeddings_per_timestep_per_batch)
            self.tf_timesteps_per_batch = tf.placeholder(tf.int32, [None])
            self.tf_saver = None

    def name(self):
        return os.path.basename(os.path.splitext(self.file)[0])

    # ----------------------[ Private Methods ]----------------------

    def _finish_init_base(self):
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
                self.file = os.path.join(self.file, self._gen_name() + ".json")
                self.tf_checkpoint_path = os.path.splitext(self.file)[0]

    # -*- coding: utf-8 -*-

    @staticmethod
    def _print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=10):
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
