# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

import codecs
from collections import defaultdict
from unidecode import unidecode
import random
import numpy as np
import json

# ============================[ Constants ]==========================

"""
These constants define an ASCII subset which will be the primary feature-set emitted by FtsCorpus
for encoding characters. Any unsupported characters will be encoded with the index of '_'. 
"""
CHAR_SUBSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-., /_"
CHAR_SUBSET_DEFAULT = CHAR_SUBSET.index("_")
CHAR_SUBSET_INDEX = defaultdict(lambda: CHAR_SUBSET_DEFAULT, ((c, i) for i, c in enumerate(CHAR_SUBSET)))

"""
This token may be used in a corpus tsv file to indicate the absence of a value.
"""
WILDCARD_TOKEN = "*"

# ============================[ FtsToken ]==========================


class FtsToken:
    """
    FtsToken is an atomic class which represents a single token
    that may occur in a Full-Text-Search (FTS) query.
    """

    # ------------------------[ Properties ]------------------------

    """
    The id represents a compact representation of the tokens global id.
    The first entry is the numeric id of the tokens class as determined by
    FtsCorpus, and the second entry is the local id of the token within the class.
    """
    id = (0, 0)

    """
    The actual string of the token
    """
    string = ""

    """
    The parent token. In the first phase of loading, this may be an id tuple like @id.
    In the second phase, this id will be converted to the actual parent, or None, if
    no valid parent is indicated.
    """
    parent = None

    """
    List of FtsTokens which are logically depending on this token as their parent.
    """
    children = []

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, class_id, token_id, parent_id_tuple, token_str):
        self.id = (class_id, token_id)
        self.string = unidecode(token_str)
        self.parent = parent_id_tuple
        self.children = []

    def recursive_parents(self, result=None):
        """
        Returns a list of this token's parent token plus parent_token.recursive_parents().
        This will return a dictionary like {<parent_token_class_id>: <FtsToken>}.
        :param result: Dictionary which should be updated with parent classes, or None, if
        a new dictionary should be allocated.
        """
        if not result:
            result = {}
        if self.parent:
            result[self.parent.id[0]] = self.parent
            self.parent.recursive_parents(result)
        return result

    def random_recursive_children(self, result=None):
        """
        Randomly returns one of @children, plus child.random_recursive_children().
        This will return a dictionary like {<child_token_class_id>: <FtsToken>}.
        :param result: Dictionary which should be updated with child classes, or None, if
        a new dictionary should be allocated.
        """
        if not result:
            result = {}
        if len(self.children) > 0:
            random_child = random.choice(self.children)
            result[random_child.id[0]] = random_child
            random_child.random_recursive_children(result)
        return result

# ===========================[ FtsGrammar ]==========================


class FtsGrammarRandomSequenceRule:

    # ------------------------[ Properties ]------------------------

    type = ""
    symbols = None

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, json_symbols, terminal_corpus, known_rules):
        assert isinstance(terminal_corpus, FtsCorpus)
        self.symbols = []
        assert isinstance(json_symbols, list)
        for json_symbol in json_symbols:
            rule_class_name = json_symbol[FtsGrammar.RULE_CLASS]
            rule_prob = json_symbol[FtsGrammar.RULE_PROBABILITY]
            rule = None
            if rule_class_name in known_rules:
                rule = known_rules[rule_class_name]
            elif rule_class_name in terminal_corpus.class_ids:
                rule = terminal_corpus.class_ids[rule_class_name]
            else:
                print("  ERROR: Failed to resolve reference to nonterminal class '{}'!".format(rule_class_name))
            if rule or isinstance(rule, int):
                self.symbols.append((rule, rule_prob))

    def generate_with_token(self, token):
        assert isinstance(token, FtsToken)
        # Collect symbols that will be generated. The given @token is definitely included.
        deck = [token.id[0]]
        available_tokens_per_class = {token.id[0]: token}
        available_tokens_per_class.update(token.recursive_parents())
        available_tokens_per_class.update(token.random_recursive_children())
        for symbol in self.symbols:
            # Non-terminals in random sequence not yet supported
            assert isinstance(symbol[0], int)
            # Do not re-evaluate fixed @token that is already in deck
            if symbol[0] != token.id[0]:
                assert symbol[0] in available_tokens_per_class
                uniform_val = random.uniform(0., 1.)
                if uniform_val <= symbol[1]:
                    deck.append(symbol[0])
        random.shuffle(deck)
        return [available_tokens_per_class[class_id] for class_id in deck]


class FtsGrammar:

    # ------------------------[ Properties ]------------------------

    ROOT_SYMBOL = "root-nonterminal"
    RULES = "rules"
    RULE_CLASS = "class"
    RULE_TYPE = "type"
    RULE_RANDOM_SEQUENCE_TYPE = "random-sequence"
    RULE_PROBABILITY = "prior"
    RULE_SYMBOLS = "symbols"

    """
    @terminal_classes describes the string/numeric identities for
    terminal classes in the grammar. Those are usually defined by
    a client FtsCorpus.  
    """
    terminal_classes = None

    """
    Dictionary which points from a nonterminal name to the rule it generates.
    A rule is tuple like (rule_type as string, [FtsGrammarRule]).
    """
    nonterminal_rules = None

    """
    Name of the nonterminal class which will be used as an entry point for sample generation.
    """
    root_nonterminal = ""

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, path, terminal_corpus):
        """
        Create a new FTS grammar from a JSON definition file.
        :param path: Path of the JSON-file which holds the grammar.
        :param terminal_corpus: FtsCorpus which provides the terminal symbols for the grammar.
        """
        with codecs.open(path) as json_grammar_file:
            print("Loading {} ...".format(path))
            json_grammar = json.load(json_grammar_file)
        self.terminal_classes = terminal_corpus.class_ids
        self.root_nonterminal = json_grammar[FtsGrammar.ROOT_SYMBOL]
        self.nonterminal_rules = dict()
        for json_rule in json_grammar[FtsGrammar.RULES]:
            new_rule = None
            new_rule_type = json_rule[FtsGrammar.RULE_TYPE]
            new_rule_class = json_rule[FtsGrammar.RULE_CLASS]

            if new_rule_type == FtsGrammar.RULE_RANDOM_SEQUENCE_TYPE:
                new_rule = FtsGrammarRandomSequenceRule(
                    json_rule[FtsGrammar.RULE_SYMBOLS],
                    terminal_corpus,
                    self.nonterminal_rules)
            else:
                print("  Unsupported rule type '{}'!".format(new_rule_type))
            if new_rule:
                self.nonterminal_rules[new_rule_class] = new_rule
                print("  Added '{}' rule for '{}'.".format(new_rule_type, new_rule_class))

    def random_phrase_with_token(self, token):
        assert isinstance(token, FtsToken)
        return self.nonterminal_rules[self.root_nonterminal].generate_with_token(token)


# ============================[ FtsCorpus ]==========================


class FtsCorpus:
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

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, path):
        self.class_ids = defaultdict(lambda: len(self.class_ids))
        self.data = defaultdict(lambda: [])
        token_for_id = {}

        with codecs.open(path) as corpus_file:
            print("Loading {} ...".format(path))
            for entry in corpus_file:
                parts = entry.strip().split("\t")
                if len(parts) >= 6:
                    class_id = self.class_ids[parts[0]]
                    token_id = int(parts[1])
                    token_str = parts[2]
                    parent_class_id = "*"
                    parent_token_id = 0
                    if parts[4] != WILDCARD_TOKEN:
                        parent_class_id = self.class_ids[parts[4]]
                        parent_token_id = int(parts[5])
                    token_for_id[(class_id, token_id)] = FtsToken(
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
        # + 1 for EOL class
        return len(CHAR_SUBSET) + len(self.class_ids) + 1

    def get_batch_and_lengths(self, batch_size, sample_grammar, epoch_leftover_indices=None, train_test_split=None):
        """
        Returns a new batch-first character feature matrix like [batch_size][sample_length][char_features].
        :param batch_size: The number of sample sequences to return.
        :param sample_grammar: The grammar to use for sample generation. Must be one of grammar.FtsGrammar.
        :param epoch_leftover_indices: The iterator to use for sample selection.
         Should be either None or previous 3rd return value.
         A (return) value of None or [] indocates the start of a new epoch.
        :param train_test_split: Unused.
        """
        assert (isinstance(sample_grammar, FtsGrammar))
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
            # Length of all tokens ...                 + White space ...      + End-of-line
            sum(len(token.string) for token in phrase_tokens) + len(phrase_tokens)-1 + 1
            for phrase_tokens in batch_phrases)
        batch_embedding_sequences = [
            self._embed(phrase_tokens, max_phrase_length)
            for phrase_tokens in batch_phrases]
        batch_lengths = np.asarray([
            len(batch_embedding_sequence)
            for batch_embedding_sequence in batch_embedding_sequences])
        print(batch_embedding_sequences)
        return np.asarray(batch_embedding_sequences, np.float32), batch_lengths, epoch_leftover_indices

    # ----------------------[ Private Methods ]----------------------

    def _embed(self, token_list, length_to_align):
        """
        Embeds a sequence of FtsToken instances into a 2D feature matrix
        like [num_characters][total_num_features_per_character()].
        """
        print(" ".join((token.string for token in token_list)))
        result = []
        for token in token_list:
            assert(isinstance(token, FtsToken))
            # Iterate over all tokens. Prepend separators if necessary.
            for char in (" " if result else "")+token.string:
                char_embedding = np.zeros(self.total_num_features_per_character())
                # Set character label
                char_embedding[CHAR_SUBSET_INDEX[char]] = 1.
                # Set class label
                char_embedding[len(CHAR_SUBSET_INDEX) + token.id[0]] = 1.
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
