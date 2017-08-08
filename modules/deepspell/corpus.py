# (C) 2017 Klebert Engineering GmbH

import codecs
from collections import defaultdict
import random

from . import grammar

class FtsToken:
    """
    FtsToken is an atomic class which represents a single token
    that may occur in a Full-Text-Search (FTS) query.
    """

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

    def __init__(self, class_id, token_id, parent_id_tuple, token_str):
        self.id = (class_id, token_id)
        self.string = token_str
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


class FtsCorpus:
    """
    FtsCorpus wraps a collection of FTS (Full-Text-Search) Tokens,
    which may serve as components of FTS queries.
    """

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

    def __init__(self, path):
        self.class_ids = defaultdict(lambda: len(self.class_ids))
        self.data = defaultdict(lambda: [])
        token_for_id = {}

        with codecs.open(path) as corpus_file:
            for entry in corpus_file:
                parts = entry.split("\t")
                if len(parts) >= 6:
                    class_id = self.class_ids[parts[0]]
                    token_id = int(parts[1])
                    token_str = parts[2]
                    parent_class_id = self.class_ids[parts[4]]
                    parent_token_id = int(self.class_ids[parts[5]])
                    token_for_id[(class_id, token_id)] = FtsToken(
                        class_id,
                        token_id,
                        (parent_class_id, parent_token_id),
                        token_str)

        for _, token in token_for_id:
            if token.parent in token_for_id:
                token.parent = token_for_id[token.parent]
                token.parent.children.append(token)
            else:
                token.parent = None

    def next_batch(self, batch_size, sample_grammar, iterator=None):
        assert(isinstance(sample_grammar, grammar.FtsGrammar))
        pass