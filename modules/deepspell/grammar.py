# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

import codecs
from unidecode import unidecode
import random
import json
import math
import numpy as np
import sys

# ==========================[ Local Imports ]========================

from . import featureset

# ============================[ Constants ]==========================

"""
This token may be used in a corpus tsv file to indicate the absence of a value.
"""
WILDCARD_TOKEN = "*"


# ============================[ FtsToken ]==========================

class DSToken:
    """
    FtsToken is an atomic class which represents a single token
    that may occur in a Full-Text-Search (FTS) query.
    """

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, class_id, token_id, parent_id_tuple, token_str):
        # This is the tokens global id:
        # * The first entry is the numeric id of the tokens class as determined by FtsCorpus.
        # * The second entry is the local id of the token within the class.
        self.id = (class_id, token_id)

        # The actual string of the token
        self.string = unidecode(token_str)

        # The parent token. In the first phase of loading, this may be an id tuple like @id.
        # In the second phase, this id will be converted to the actual parent, or None, if
        # no valid parent is indicated.
        self.parent = parent_id_tuple

        # List of FtsTokens which are logically depending on this token as their parent.
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


# ==================[ FtsGrammarRandomSequenceRule ]=================

class DSGrammarRandomSequenceRule:

    type = "random-sequence"

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, json_symbols, terminal_featureset, known_rules):
        self.symbols = []
        assert isinstance(json_symbols, list)
        assert isinstance(terminal_featureset, featureset.DSFeatureSet)
        for json_symbol in json_symbols:
            rule_class_name = json_symbol[DSGrammar.RULE_CLASS]
            rule_prob = json_symbol[DSGrammar.RULE_PROBABILITY]
            rule = None
            if rule_class_name in known_rules:
                rule = known_rules[rule_class_name]
            elif rule_class_name in terminal_featureset.class_ids:
                rule = terminal_featureset.class_ids[rule_class_name]
            else:
                print("  ERROR: Failed to resolve reference to nonterminal class '{}'!".format(rule_class_name))
            if rule or isinstance(rule, int):
                self.symbols.append((rule, rule_prob))

    def generate_with_token(self, token, available_tokens_per_class=None, deck=None):
        assert isinstance(token, DSToken)
        is_root_call = False
        if not deck:
            is_root_call = True
            # Collect symbols that will be generated. The given @token is definitely included.
            assert not available_tokens_per_class
            deck = [token.id[0]]
            available_tokens_per_class = {token.id[0]: token}
            available_tokens_per_class.update(token.recursive_parents())
            available_tokens_per_class.update(token.random_recursive_children())
        for symbol, symbol_prior in self.symbols:
            symbol_prob = random.uniform(0., 1.)
            # Do not re-evaluate fixed @token that is already in deck
            if isinstance(symbol, DSGrammarRandomSequenceRule):
                if symbol_prob <= symbol_prior or symbol.has_terminal_recursive(token.id[0]):
                    symbol.generate_with_token(token, available_tokens_per_class, deck)
            else:
                assert isinstance(symbol, int)
                if symbol != token.id[0] and symbol in available_tokens_per_class:
                    if symbol_prob <= symbol_prior:
                        deck.append(symbol)
        if is_root_call:
            random.shuffle(deck)
            return [available_tokens_per_class[class_id] for class_id in deck]

    def has_terminal_recursive(self, terminal_class):
        for symbol, _ in self.symbols:
            if symbol == terminal_class:
                return True
            elif isinstance(symbol, DSGrammarRandomSequenceRule):
                return symbol.has_terminal_recursive(terminal_class)
        return False


# ===========================[ FtsGrammar ]==========================

class DSGrammar:

    ROOT_SYMBOL = "root-nonterminal"
    RULES = "rules"
    CORRUPTION = "corruption"
    CORRUPTION_MEAN = "mean"
    CORRUPTION_STDDEV = "stddev"
    RULE_CLASS = "class"
    RULE_TYPE = "type"
    RULE_RANDOM_SEQUENCE_TYPE = "random-sequence"
    RULE_PROBABILITY = "prior"
    RULE_SYMBOLS = "symbols"

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, path, terminal_featureset):
        """
        Create a new FTS grammar from a JSON definition file.
        :param path: Path of the JSON-file which holds the grammar.
        :param terminal_featureset: DSFeatureSet which provides the terminal symbols for the grammar.
        """
        assert isinstance(terminal_featureset, featureset.DSFeatureSet)
        with codecs.open(path) as json_grammar_file:
            print("Loading {} ...".format(path))
            json_grammar = json.load(json_grammar_file)

        # Describes the string/numeric identities for terminal classes in the grammar.
        # Those are usually defined by a client FtsCorpus.
        self.terminal_classes = terminal_featureset.class_ids

        # Dictionary which points from a nonterminal name to the rule it generates.
        # A rule is tuple like (rule_type as string, [FtsGrammarRule]).
        self.root_nonterminal = json_grammar[DSGrammar.ROOT_SYMBOL]

        # Name of the nonterminal class which will be used as an entry point for sample generation.
        self.nonterminal_rules = dict()

        # Normal distribution over the number of errors that should be inserted into the output tokens.
        self.corruption_dist_mean = json_grammar[DSGrammar.CORRUPTION][DSGrammar.CORRUPTION_MEAN]
        self.corruption_dist_stddev = json_grammar[DSGrammar.CORRUPTION][DSGrammar.CORRUPTION_STDDEV]

        for json_rule in json_grammar[DSGrammar.RULES]:
            new_rule = None
            new_rule_type = json_rule[DSGrammar.RULE_TYPE]
            new_rule_class = json_rule[DSGrammar.RULE_CLASS]

            if new_rule_type == DSGrammar.RULE_RANDOM_SEQUENCE_TYPE:
                new_rule = DSGrammarRandomSequenceRule(
                    json_rule[DSGrammar.RULE_SYMBOLS],
                    terminal_featureset,
                    self.nonterminal_rules)
            else:
                print("  Unsupported rule type '{}'!".format(new_rule_type))
            if new_rule:
                self.nonterminal_rules[new_rule_class] = new_rule
                print("  Added '{}' rule for '{}'.".format(new_rule_type, new_rule_class))

    def random_phrase_with_token(self, token):
        assert isinstance(token, DSToken)
        return self.nonterminal_rules[self.root_nonterminal].generate_with_token(token)

    def corrupt(self, string_to_corrupt):
        """
        Corrupts a string with deletions, switches, insertions and substitutions `n` times,
        where `n` is the floor of a number that is drawn from the normal distribution given by
        `self.corruption_dist_mean` and `self.corruption_dist_stddev`. Note, that `n` is also
        capped by `self.max_corruptions()`.
        :param string_to_corrupt: The string that should be corrupted.
        :return: The corrupted string.
        """
        def delete_char(s):
            # sys.stdout.write(" del")
            pos = int(math.floor(random.uniform(0, len(s))))
            return s[:pos]+s[pos+1:]

        def subst_char(s):
            # sys.stdout.write(" subst")
            pos = int(math.floor(random.uniform(0, len(s))))
            ch = chr(ord('a') + int(random.uniform(0, 26)))
            return s[:pos] + ch + s[pos+1:]

        def switch_char(s):
            # sys.stdout.write(" switch")
            pos1 = int(math.floor(random.uniform(0, len(s)-1)))
            pos2 = int(math.floor(random.uniform(pos1+1, len(s))))
            return s[:pos1] + s[pos2] + s[pos1+1:pos2] + s[pos1] + s[pos2+1:]

        def insert_char(s):
            # sys.stdout.write(" ins")
            pos = int(math.floor(random.uniform(0, len(s)+1)))
            ch = chr(ord('a') + int(random.uniform(0, 26)))
            return s[:pos] + ch + s[pos:]

        n = min(
            self.max_corruptions(),
            int(np.floor(np.random.normal(self.corruption_dist_mean, self.corruption_dist_stddev))))
        # print("Corrupting", string_to_corrupt, "...", end="")
        for i in range(n):
            # don't corrupt below 2 characters
            if len(string_to_corrupt) < 3:
                break
            corruption_function = random.choice((delete_char, subst_char, switch_char, insert_char))
            string_to_corrupt = corruption_function(string_to_corrupt)
        # print(" =>", string_to_corrupt)
        return string_to_corrupt

    def max_corruptions(self):
        return int(self.corruption_dist_mean + self.corruption_dist_stddev*5.)
