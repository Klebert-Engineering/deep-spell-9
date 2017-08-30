# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

import codecs
from collections import defaultdict
import random
import numpy as np

# ==========================[ Local Imports ]========================

from . import grammar

# ============================[ Constants ]==========================

"""
These constants define an ASCII subset which will be the primary feature-set emitted by FtsCorpus
for encoding characters. Any unsupported characters will be encoded with the index of '_'. 
"""
CHAR_SUBSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-., /_"
CHAR_SUBSET_DEFAULT = CHAR_SUBSET.index("_")
CHAR_SUBSET_INDEX = defaultdict(lambda: CHAR_SUBSET_DEFAULT, ((c, i) for i, c in enumerate(CHAR_SUBSET)))

# ============================[ FtsCorpus ]==========================


class DSCorpus:
    """
    FtsCorpus wraps a collection of FTS (Full-Text-Search) Tokens,
    which may serve as components of FTS queries.
    """

    # ------------------------[ Properties ]------------------------

    """
    @class_ids is a dictionary like:
    { <class_name_string>: <class_id> }
    """
    class_ids = None

    """
    @data is a dictionary like:
    { <class_id>: [<FtsToken>] }
    """
    data = None

    """
    @name An arbitrary short identifier for the corpus.
    """
    name = ""

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, path, name):
        self.name = name
        self.class_ids = defaultdict(lambda: len(self.class_ids))
        self.data = defaultdict(lambda: [])
        token_for_id = {}

        with codecs.open(path, encoding='utf-8') as corpus_file:
            print("Loading {} ...".format(path))
            for entry in corpus_file:
                parts = entry.strip().split("\t")
                if len(parts) >= 6:
                    class_id = self.class_ids[parts[0]]
                    token_id = int(parts[1])
                    token_str = parts[2]
                    parent_class_id = "*"
                    parent_token_id = 0
                    if parts[4] != grammar.WILDCARD_TOKEN:
                        parent_class_id = self.class_ids[parts[4]]
                        parent_token_id = int(parts[5])
                    token_for_id[(class_id, token_id)] = grammar.DSToken(
                        class_id,
                        token_id,
                        (parent_class_id, parent_token_id),
                        token_str)

        print("  Read {} tokens:".format(len(token_for_id)))
        for (class_id, _), token in token_for_id.items():
            self.data[class_id].append(token)
            if token.parent in token_for_id:
                token.parent = token_for_id[token.parent]
                token.parent.children.append(token)
            else:
                token.parent = None

        for class_name, class_id in self.class_ids.items():
            print("  * {} tokens for class '{}'".format(len(self.data[class_id]), class_name))

    def total_num_features_per_character(self):
        return self.total_num_lexical_features_per_character() + self.total_num_logical_features_per_character()

    @staticmethod
    def total_num_lexical_features_per_character():
        return len(CHAR_SUBSET)

    def total_num_logical_features_per_character(self):
        return len(self.class_ids) + 1  # + 1 for EOL class

    def get_batch_and_lengths(self, batch_size, sample_grammar, epoch_leftover_indices=None, train_test_split=None):
        """
        Returns a new batch-first character feature matrix like [batch_size][sample_length][char_features].
        :param batch_size: The number of sample sequences to return.
        :param sample_grammar: The grammar to use for sample generation. Must be one of grammar.FtsGrammar.
        :param epoch_leftover_indices: The iterator to use for sample selection.
         Should be either None or previous 3rd return value.
         A (return) value of None or [] indicates the start of a new epoch.
        :param train_test_split: Unused.
        """
        assert (isinstance(sample_grammar, grammar.DSGrammar))
        # Make sure that training document order is randomized
        if not epoch_leftover_indices:
            epoch_leftover_indices = [
                (class_id, i)
                for class_id, class_tokens in self.data.items()
                for i in range(len(class_tokens))]
            random.shuffle(epoch_leftover_indices)
        # First, collect all the texts that will be put into the batch
        batch_token_indices = epoch_leftover_indices[:batch_size]
        epoch_leftover_indices = epoch_leftover_indices[batch_size:]
        # Compile the lengths of the token sequences of the selected examples
        batch_phrases = [
            sample_grammar.random_phrase_with_token(self.data[token_id[0]][token_id[1]])
            for token_id in batch_token_indices]
        # Find the longest phrase, such that all lines in the output matrix can be length-aligned
        max_phrase_length = max(
            # Length of all tokens ...                        + White space ...      + End-of-line
            sum(len(token.string) for token in phrase_tokens) + len(phrase_tokens)-1 + 1
            for phrase_tokens in batch_phrases)
        batch_embedding_sequences = np.asarray([
            self._embed(phrase_tokens, max_phrase_length)
            for phrase_tokens in batch_phrases], np.float32)
        batch_lengths = np.asarray([
            len(batch_embedding_sequence)
            for batch_embedding_sequence in batch_embedding_sequences])
        return batch_embedding_sequences, batch_lengths, epoch_leftover_indices

    # ----------------------[ Private Methods ]----------------------

    def _embed(self, token_list, length_to_align):
        """
        Embeds a sequence of FtsToken instances into a 2D feature matrix
        like [num_characters][total_num_features_per_character()].
        """
        result = []
        for token in token_list:
            assert(isinstance(token, grammar.DSToken))
            # Iterate over all tokens. Prepend whitespace if necessary.
            for char in (" " if result else "")+token.string:
                char_embedding = np.zeros(self.total_num_features_per_character())
                # Set character label
                char_embedding[CHAR_SUBSET_INDEX[char]] = 1.
                # Set class label
                char_embedding[self.total_num_lexical_features_per_character() + token.id[0]] = 1.
                result.append(char_embedding)
        # Append EOL char
        char_embedding = np.zeros(self.total_num_features_per_character())
        char_embedding[len(char_embedding)-1] = 1.
        result.append(char_embedding)
        # Align output length
        assert len(result) <= length_to_align
        while len(result) < length_to_align:
            result.append(np.zeros(self.total_num_features_per_character()))
        return np.asarray(result, np.float32)
