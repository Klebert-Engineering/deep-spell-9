import json
import os

# ====================[ Abstract Predictor Interface ]===================


class FtsPredictor:

    # ------------------------[ Properties ]------------------------

    file = ""

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, embedding_model, file_or_folder="", kwargs_to_update=None):
        """
        Instantiate or restore a Representation Model.
        :param file_or_folder: [optional] A file from which the model
         should be restored, or to which it should be saved.
        :param kwargs_to_update: [optional] A kwargs dictionary
         that should be updated with information stored in JSON-format
         under @file.
        """
        assert isinstance(file_or_folder, str)
        self.file = file_or_folder
        if os.path.isfile(file_or_folder):
            print("Loading model '{}'...".format(file_or_folder))
            assert os.path.splitext(file_or_folder)[1] == ".json"
            kwargs_to_update.update(json.load(open(file_or_folder, "r")))
        elif file_or_folder:
            assert os.path.isdir(file_or_folder)

    def store(self, file):
        """
        Store this model under a certain file path.
        :param file: The file path under which it should be stored.
        """
        pass

    def train(self, corpus, train_test_split=None):
        """
        Train this model with documents from a given corpus.
        :param corpus: The corpus with documents to train from.
         Should be an instance of iromir.corpus.Corpus.
        :param train_test_split: [optional] A 2-tuple that indicates the ratio
         for the train-test split of the corpus.
        """
        pass
