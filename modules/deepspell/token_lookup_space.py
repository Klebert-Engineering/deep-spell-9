# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

import codecs
import pickle
import os

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("WARNING: SciPy not installed!")
    cKDTree = None
    pass

# ==========================[ Local Imports ]========================

from deepspell.models import encoder


# =======================[ DSTokenLookupSpace ]======================

class DSTokenLookupSpace:
    """
    `DSTokenLookupSpace` represents a vector space of NDS tokens, where
    tokens are mapped to vectors by a `DSVariationalLstmAutoEncoder`.
    """

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, model, path, encode_batch_size=16384):
        """
        Load a token lookup space, or create a new one from a corpus of tokens and a token encoding model.
        :param model: The model which should be used to encode tokens into vectors.
        :param path: Either path prefix where the <path>.tokens and <path>.kdtree files for this lookup space
         should be loaded from/stored, or path to .tsv file for corpus that should be encoded.
        """
        if not cKDTree:
            print("WARNING: SciPy not installed!")
            return
        i = 0
        assert isinstance(model, encoder.DSVariationalLstmAutoEncoder)
        assert isinstance(path, str)
        self.model = model
        token_file_path = os.path.splitext(path)[0] + ".tokens"
        kdtree_file_path = os.path.splitext(path)[0] + ".kdtree"
        if not os.path.exists(token_file_path) or not os.path.exists(kdtree_file_path):
            print("Creating new DSTokenLookupSpace under '{}' for model '{}' and corpus '{}'!".format(
                path, model.name(), path))
            self.tokens, self.kdtree = model.encode_corpus(path, encode_batch_size)
            print("Dumping tokens to '{}' ...".format(token_file_path))
            with codecs.open(token_file_path, "w") as token_output_file:
                for token in self.tokens:
                    token_output_file.write(token+"\n")
            print("  ... done.")
            print("Dumping tree to '{}' ...".format(kdtree_file_path))
            with codecs.open(kdtree_file_path, "wb") as kdtree_output_file:
                pickle.dump(self.kdtree, kdtree_output_file)
            print("  ... done.")
        else:
            self.tokens = [token.strip() for token in open(token_file_path, "r")]
            self.kdtree = pickle.load(open(kdtree_file_path, "rb"))

    def match(self, token, k=3):
        """
        Obtain a list of @k nearest neighbors to the given @token's vector in this vector space.
        :param token: The token string which should be encoded, and whose nearset neighbor should be retrieved.
        :param k: The number of correction suggestions that should be retrieved.
        :return: An ascendingly ordered list of @k (token, distance) pairs.
        """
        lookup_vec = self.model.encode(token)
        query_result_distance, query_result_indices = self.kdtree.query(lookup_vec, k=k)
        return [(self.tokens[i], d) for i, d in zip(query_result_indices, query_result_distance)]
