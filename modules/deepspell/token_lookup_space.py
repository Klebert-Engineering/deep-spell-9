# (C) 2018-present Klebert Engineering

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

    def match(self, token, k=3, sort_by_dameraulevenshtein=True):
        """
        Obtain a list of @k nearest neighbors to the given @token's vector in this vector space.
        :param token: The token string which should be encoded, and whose nearset neighbor should be retrieved.
        :param k: The number of correction suggestions that should be retrieved.
        :param sort_by_dameraulevenshtein: Flag to indicate, whether the spell results should be sorted by their
         damerau-levenshtein distance from the input in place of the euclidean distance.
        :return: An ascendingly ordered list of @k (token, distance) pairs.
        """
        lookup_vec = self.model.encode(token)
        query_result_distance, query_result_indices = self.kdtree.query(lookup_vec, k=k*4)
        return sorted(
            [(self.tokens[i], d) for i, d in zip(query_result_indices, query_result_distance)],
            key=lambda token_and_distance: self._dameraulevenshtein(token, token_and_distance[0]))[:k]

    # ---------------------[ Private Methods ]---------------------

    @staticmethod
    def _dameraulevenshtein(seq1, seq2):
        """Calculate the Damerau-Levenshtein distance between sequences.

        This method has not been modified from the original.
        Source: http://mwh.geek.nz/2009/04/26/python-damerau-levenshtein-distance/

        This distance is the number of additions, deletions, substitutions,
        and transpositions needed to transform the first sequence into the
        second. Although generally used with strings, any sequences of
        comparable objects will work.

        Transpositions are exchanges of *consecutive* characters; all other
        operations are self-explanatory.

        This implementation is O(N*M) time and O(M) space, for N and M the
        lengths of the two sequences.

        >>> dameraulevenshtein('ba', 'abc')
        2
        >>> dameraulevenshtein('fee', 'deed')
        2

        It works with arbitrary sequences too:
        >>> dameraulevenshtein('abcd', ['b', 'a', 'c', 'd', 'e'])
        2
        """
        # codesnippet:D0DE4716-B6E6-4161-9219-2903BF8F547F
        # Conceptually, this is based on a len(seq1) + 1 * len(seq2) + 1 matrix.
        # However, only the current and two previous rows are needed at once,
        # so we only store those.
        oneago = None
        thisrow = list(range(1, len(seq2) + 1)) + [0]
        for x in range(len(seq1)):
            # Python lists wrap around for negative indices, so put the
            # leftmost column at the *end* of the list. This matches with
            # the zero-indexed strings and saves extra calculation.
            twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
            for y in range(len(seq2)):
                delcost = oneago[y] + 1
                addcost = thisrow[y - 1] + 1
                subcost = oneago[y - 1] + (seq1[x] != seq2[y])
                thisrow[y] = min(delcost, addcost, subcost)
                # This block deals with transpositions
                if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                    and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                    thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
        return thisrow[len(seq2) - 1]
