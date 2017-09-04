# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

import codecs
from collections import defaultdict
import random
import numpy as np
import math

# ==========================[ Local Imports ]========================

from . import grammar

# ============================[ Constants ]==========================

"""
These constants define an ASCII subset which will be the primary feature-set emitted by FtsCorpus
for encoding characters. The set contains the following special characters:
* Any unsupported characters will be encoded with the index of '_'.
* Any characters of the EOL class will be encoded with '$'
"""
CHAR_SUBSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-., /_$"
CHAR_SUBSET_DEFAULT = CHAR_SUBSET.index("_")
CHAR_SUBSET_EOL = CHAR_SUBSET.index("$")
CHAR_SUBSET_INDEX = defaultdict(lambda: CHAR_SUBSET_DEFAULT, ((c, i) for i, c in enumerate(CHAR_SUBSET)))

# ============================[ FtsCorpus ]==========================


class DSCorpus:
    """
    FtsCorpus wraps a collection of FTS (Full-Text-Search) Tokens,
    which may serve as components of FTS queries.
    """

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, path, name):
        self.name = name
        # -- class_ids is a dictionary like { <class_name_string>: <class_id> }
        self.class_ids = defaultdict(lambda: len(self.class_ids))
        # -- data is a dictionary like: { <class_id>: [<FtsToken>] }
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

        self.eol_class_id = self.class_ids["EOL"]

    def total_num_features_per_character(self):
        return self.num_lexical_features_per_character() + self.num_logical_features_per_character()

    @staticmethod
    def num_lexical_features_per_character():
        return len(CHAR_SUBSET)

    def num_logical_features_per_character(self):
        return len(self.class_ids)

    def get_batch_and_lengths(self,
                              batch_size,
                              sample_grammar,
                              epoch_leftover_indices=None,
                              train_test_split=None,
                              min_num_chars_truncate=-1):
        """
        Returns a new batch-first character-feature matrix like [batch_size][sample_length][char_features].
        :param batch_size: The number of sample sequences to return.
        :param sample_grammar: The grammar to use for sample generation. Must be one of grammar.FtsGrammar.
        :param epoch_leftover_indices: The iterator to use for sample selection.
         Should be either None or previous 3rd return value.
         A (return) value of None or [] indicates the start of a new epoch.
        :param train_test_split: Unused.
        :param min_num_chars_truncate: Randomly truncate the generated samples
         to a length of <min_num_chars_truncate>.
         For example, let min_num_chars_truncate=4 and sample="Los Angeles California".
         The sample will be randomly truncated to a length between 4 and 22, so
         it may become one of {"Los ", "Los A", "Los An", .., "Los Angeles California"}.
         This is useful to train the discriminator network to recognize categories from incomplete samples.
         Truncation will be omitted entirely if min_num_chars_truncate<0.
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
        batch_lengths = np.asarray([
            self._token_sequence_length(phrase_tokens)
            for phrase_tokens in batch_phrases])
        max_phrase_length = max(batch_lengths)
        batch_embedding_sequences = np.asarray([
            self._embed(phrase_tokens, max_phrase_length, min_num_chars_truncate)
            for phrase_tokens in batch_phrases], np.float32)
        return batch_embedding_sequences, batch_lengths, epoch_leftover_indices

    def class_name_for_id(self, id):
        """
        Returns the name of the token class for the given integer class id, or None,
        if no class with that id was found. Asserts that id is a unique value in self.class_ids.
        :param id: The integer id of class name to be retrieved.
        :return: The class name, or None, if no class with the given id exists.
        """
        result = [name for name, name_id in self.class_ids.items() if name_id == id]
        assert len(result) <= 1
        return result[0] if result else None

    def embed_characters(self, characters, classes=None):
        """
        Embeds a character sequence with an optional class annotation into a 2D-matrix.
        :param characters: Arbitrary string. If it does not end in the EOL-character '$',
         the EOL-character will be appended automatically.
        :param classes: Optional list of terminal token class names. Must be empty or iterable
         of length len(prefix_chars). If empty, no class features will be written into the result matrix.
        :return: The embedded feature matrix with shape
         (len(characters), self.total_num_features_per_character()). Note, that all logical
         features will be set to zero if classes is empty/None.
        """
        assert not classes or len(classes) == len(characters)
        result = []
        if not characters[-1] == CHAR_SUBSET[CHAR_SUBSET_EOL]:
            characters += CHAR_SUBSET[CHAR_SUBSET_EOL]
            if classes:
                classes += ["EOL"]
        for i, char in enumerate(characters):
            char_id = CHAR_SUBSET_INDEX[char]
            char_embedding = np.zeros(self.total_num_features_per_character())
            # Set character label
            char_embedding[char_id] = 1.
            # Set class label
            if classes:
                class_id = self.class_ids[classes[i]]
                char_embedding[self.num_lexical_features_per_character() + class_id] = 1.
            result.append(char_embedding)
        return np.asarray(result, np.float32)

    # ----------------------[ Private Methods ]----------------------

    def _embed(self, token_list, length_to_align, min_chars_truncate):
        """
        Embeds a sequence of FtsToken instances into a 2D feature matrix
        like [num_characters][total_num_features_per_character()].
        """
        result = []
        char_class_seq = [
            (char, token.id[0])
            for i, token in enumerate(token_list)
            for char in (" " if i > 0 else "")+token.string]

        if len(char_class_seq) > min_chars_truncate and min_chars_truncate >= 0:
            truncation_point = (math.ceil(random.uniform(0., 1.) * (len(char_class_seq) - min_chars_truncate)) +
                                min_chars_truncate)
            char_class_seq = char_class_seq[:truncation_point]

        for char, token_class in char_class_seq:
            # Iterate over all tokens. Prepend whitespace if necessary.
            char_embedding = np.zeros(self.total_num_features_per_character())
            # Set character label
            char_embedding[CHAR_SUBSET_INDEX[char]] = 1.
            # Set class label
            char_embedding[self.num_lexical_features_per_character() + token_class] = 1.
            result.append(char_embedding)

        # Align output length by padding with EOL chars
        assert len(result) < length_to_align
        while len(result) < length_to_align:
            char_embedding = np.zeros(self.total_num_features_per_character())
            char_embedding[CHAR_SUBSET_EOL] = 1.
            char_embedding[self.num_lexical_features_per_character() + self.eol_class_id] = 1.
            result.append(char_embedding)
        return np.asarray(result, np.float32)

    @staticmethod
    def _token_sequence_length(tokens):
        """
        Calculates the length of the String that would result if all the strings
        in the given list of DSToken objects were concatenated, including 1 separator
        between all tokens and a final End-Of-Line character.
        :param tokens: List of DSToken objects.
        """
        return (
            sum(len(token.string) for token in tokens) +  # Length of all tokens
            len(tokens) - 1 +                             # White space
            1                                             # End-of-line
        )
