# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

import codecs
from collections import defaultdict
import random
import numpy as np

# ==========================[ Local Imports ]========================

from . import grammar
from . import featureset


# ============================[ DSCorpus ]===========================

class DSCorpus:
    """
    FtsCorpus wraps a collection of FTS (Full-Text-Search) Tokens,
    which may serve as components of FTS queries.
    """

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, path, name, lowercase=False):
        self.name = name
        # -- class_ids is a dictionary like { <class_name_string>: <class_id> }
        class_ids = defaultdict(lambda: len(class_ids))
        # -- data is a dictionary like: { <class_id>: [<FtsToken>] }
        self.data = defaultdict(lambda: [])
        token_for_id = {}

        with codecs.open(path, encoding='utf-8') as corpus_file:
            print("Loading {} ...".format(path))
            for entry in corpus_file:
                parts = entry.strip().split("\t")
                if len(parts) >= 6:
                    class_id = class_ids[parts[0]]
                    token_id = int(parts[1])
                    token_str = parts[2].lower() if lowercase else parts[2]
                    parent_class_id = "*"
                    parent_token_id = 0
                    if parts[4] != grammar.WILDCARD_TOKEN:
                        parent_class_id = class_ids[parts[4]]
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

        for class_name, class_id in class_ids.items():
            print("  * {} tokens for class '{}'".format(len(self.data[class_id]), class_name))

        # -- Create featureset from the gathered class ids
        self.featureset = featureset.DSFeatureSet(
            classes=class_ids,
            charset=featureset.DSFeatureSet.LOWER_CASE_CHARSET if lowercase else featureset.DSFeatureSet.FULL_CASE_CHARSET)

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
            self.featureset.embed_tokens(phrase_tokens, max_phrase_length, min_num_chars_truncate)
            for phrase_tokens in batch_phrases], np.float32)
        return batch_embedding_sequences, batch_lengths, epoch_leftover_indices

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
