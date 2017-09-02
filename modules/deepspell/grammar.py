# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

import codecs
from unidecode import unidecode
import random
import json

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

    # ------------------------[ Properties ]------------------------

    """
    This is the tokens global id:
    * The first entry is the numeric id of the tokens class as determined by FtsCorpus.
    * The second entry is the local id of the token within the class.
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


# ==================[ FtsGrammarRandomSequenceRule ]=================

class DSGrammarRandomSequenceRule:

    # ------------------------[ Properties ]------------------------

    type = ""
    symbols = None

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, json_symbols, terminal_corpus, known_rules):
        self.symbols = []
        assert isinstance(json_symbols, list)
        for json_symbol in json_symbols:
            rule_class_name = json_symbol[DSGrammar.RULE_CLASS]
            rule_prob = json_symbol[DSGrammar.RULE_PROBABILITY]
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
        assert isinstance(token, DSToken)
        # Collect symbols that will be generated. The given @token is definitely included.
        deck = [token.id[0]]
        available_tokens_per_class = {token.id[0]: token}
        available_tokens_per_class.update(token.recursive_parents())
        available_tokens_per_class.update(token.random_recursive_children())
        for symbol in self.symbols:
            # Non-terminals in random sequence not yet supported
            assert isinstance(symbol[0], int)
            # Do not re-evaluate fixed @token that is already in deck
            if symbol[0] != token.id[0] and symbol[0] in available_tokens_per_class:
                uniform_val = random.uniform(0., 1.)
                if uniform_val <= symbol[1]:
                    deck.append(symbol[0])
        random.shuffle(deck)
        return [available_tokens_per_class[class_id] for class_id in deck]


# ===========================[ FtsGrammar ]==========================

class DSGrammar:

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
        self.root_nonterminal = json_grammar[DSGrammar.ROOT_SYMBOL]
        self.nonterminal_rules = dict()
        for json_rule in json_grammar[DSGrammar.RULES]:
            new_rule = None
            new_rule_type = json_rule[DSGrammar.RULE_TYPE]
            new_rule_class = json_rule[DSGrammar.RULE_CLASS]

            if new_rule_type == DSGrammar.RULE_RANDOM_SEQUENCE_TYPE:
                new_rule = DSGrammarRandomSequenceRule(
                    json_rule[DSGrammar.RULE_SYMBOLS],
                    terminal_corpus,
                    self.nonterminal_rules)
            else:
                print("  Unsupported rule type '{}'!".format(new_rule_type))
            if new_rule:
                self.nonterminal_rules[new_rule_class] = new_rule
                print("  Added '{}' rule for '{}'.".format(new_rule_type, new_rule_class))

    def random_phrase_with_token(self, token):
        assert isinstance(token, DSToken)
        return self.nonterminal_rules[self.root_nonterminal].generate_with_token(token)
