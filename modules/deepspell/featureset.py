# (C) 2017 Klebert Engineering GmbH

# =============================[ Imports ]===========================

from collections import defaultdict
import random
import numpy as np
import math


# ==========================[ DSFeatureSet ]=========================

class DSFeatureSet:
    """
    This class encapsulates the lexical and logical feature-sets that will be used to encode
    tokens into 2-hot vectors. These feature-vectors consist of two one-hot parts:
    * The lexical featureset defines an ASCII subset for encoding characters. The set also
      contains the following special characters:
      - Any unsupported characters will be encoded with the index of self.unk_char ('_').
      - Any characters of the EOL class will be encoded with self.eol_char ('$')
    * The logical featureset indicates the semantic appropriation of each character with respect
      to its token. For the NDS use-case, this means that the logical feature encodes the FTS
      token type (like CITY, COUNTRY, ROAD etc.).
    """

    LOWER_CASE_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789-., /_$^"
    FULL_CASE_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + LOWER_CASE_CHARSET

    def __init__(self,
                 classes=None,
                 charset=LOWER_CASE_CHARSET,
                 bol_char="^",
                 eol_char="$",
                 unk_char="_",
                 eol_class_name="EOL"):

        assert not classes or isinstance(classes, dict)
        self.class_ids = classes
        self.charset = charset
        self.charset_unk_index = charset.index(unk_char)
        if bol_char in self.charset:
            self.charset_bol_index = charset.index(bol_char)
        else:
            self.charset_bol_index = 0
        self.charset_eol_index = charset.index(eol_char)
        self.charset_index = defaultdict(lambda: self.charset_unk_index, ((c, i) for i, c in enumerate(charset)))
        self.bol_char = bol_char
        self.eol_char = eol_char
        self.unk_char = unk_char
        self.eol_class_name = eol_class_name
        self.eol_class_id = -1
        if self.class_ids:
            self.eol_class_id = self.class_ids[self.eol_class_name]

    def is_compatible(self, other_featureset):
        """
        Determines whether a model that was trained with this featureset could also
        be used with <other_featureset>. The constraints are that self.charset equals
        other_featureset.charset, and self.class_ids is None or self.class_ids equals other_featureset.classes.
        :param other_featureset: The featureset which is supposed to be tested for
         `self`-sufficiency. Note that this method is not commutative: other_featureset may be
         sufficient for self, but self not sufficient for other_featureset, if other_featureset
         owns well-defined non-None class_ids but self does not.
        :return: True for compatibility, False otherwise.
        """
        assert isinstance(other_featureset, DSFeatureSet)
        return (
            self.charset == other_featureset.charset and
            self.eol_char == other_featureset.eol_char and
            self.bol_char == other_featureset.bol_char and
            self.unk_char == other_featureset.unk_char and
            self.eol_class_name == other_featureset.eol_class_name and
            (not self.class_ids or self.class_ids == other_featureset.class_ids)
        )

    def adapt_logical_features(self, other_featureset):
        """
        Adapt class ids from other_featureset. This will assert for `self.is_compatible(other_featureset)`.
        :param other_featureset: The featureset whose class ids should be adapted.
        """
        assert self.is_compatible(other_featureset)
        self.class_ids = other_featureset.class_ids
        self.eol_class_id = other_featureset.eol_class_id

    def as_dict(self):
        """
        :return: Returns a dictionary of all relevant properties, such that a DSFeatureSet
         may be constructed via DSFeatureSet(**other_featureset.as_dict()).
        """
        return {
            "charset": self.charset,
            "classes": self.class_ids,
            "bol_char": self.bol_char,
            "eol_char": self.eol_char,
            "unk_char": self.unk_char,
            "eol_class_name": self.eol_class_name
        }

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

    def total_num_features_per_character(self):
        return self.num_lexical_features() + self.num_logical_features()

    def num_lexical_features(self):
        return len(self.charset)

    def num_logical_features(self):
        return len(self.class_ids)

    def embed_characters(self, characters, classes=None, append_eol=True):
        """
        Embeds a character sequence with an optional class annotation into a 2D-matrix.
        :param characters: Arbitrary string. If it does not end in the EOL-character '$',
         the EOL-character will be appended automatically.
        :param classes: Optional list of terminal token class names or ids. Must be empty or iterable
         of length len(prefix_chars). If empty, no class features will be written into the result matrix.
        :param append_eol: Flag to determine whether an EOL character should eb appended
         to <characters> (and classes if classes is not None).
        :return: The embedded feature matrix with shape
         (len(characters), self.total_num_features_per_character()). Note, that all logical
         features will be set to zero if classes is empty/None.
        """
        assert not classes or len(classes) == len(characters)
        result = []
        if append_eol and characters[-1] != self.eol_char:
            characters += self.eol_char
            if classes:
                classes += [self.eol_class_name]
        for i, char in enumerate(characters):
            char_id = self.charset_index[char]
            char_embedding = np.zeros(self.total_num_features_per_character())
            # Set character label
            char_embedding[char_id] = 1.
            # Set class label
            if classes:
                if isinstance(classes[i], int):
                    class_id = classes[i]
                else:
                    class_id = self.class_ids[classes[i]]
                char_embedding[self.num_lexical_features() + class_id] = 1.
            result.append(char_embedding)
        return np.asarray(result, np.float32)

    def embed_tokens(self, token_list, length_to_align, min_chars_truncate, corruption_grammar=None, embed_with_class=True):
        """
        Embeds a sequence of FtsToken instances into 2D feature matrices.
        => Note, that applying both truncation and corruption is not yet supported.
        :return: Returns four values:
        1. A 2D charcter-embedding feature matrix of shape [length_to_align][total_num_features]
        2. A scalar int with the true length of the sample
        3. If corruption_grammar is not None:
         A corrupted version of ret.val. (1) with ONLY LEXICAL FEATURES, None otherwise.
        4. If corruption_grammar is not None:
         A scalar int with the true length of the corrupted sample, otherwise None.
        """
        result = []  # 1st return value
        corrupted_result = None  # 3rd return value
        true_corrupted_sample_length = None
        char_class_seq = [
            (char, token.id[0])
            for i, token in enumerate(token_list)
            for char in (" " if i > 0 else "")+token.string]
        corrupted_char_class_seq = []
        embedding_size = self.total_num_features_per_character() if embed_with_class else self.num_lexical_features()

        if corruption_grammar:
            assert min_chars_truncate < 0
            corrupted_result = []
            corrupted_char_class_seq = [
                (char, token.id[0])
                for i, token in enumerate(token_list)
                for char in (" " if i > 0 else "")+corruption_grammar.corrupt(token.string)]
        elif (len(char_class_seq) > min_chars_truncate) and (min_chars_truncate >= 0):
            corrupted_char_class_seq = []
            truncation_point = (
                math.ceil(random.uniform(0., 1.) * (len(char_class_seq) - min_chars_truncate)) +
                min_chars_truncate)
            char_class_seq = char_class_seq[:truncation_point]

        for char, token_class in char_class_seq:
            char_embedding = np.zeros(embedding_size)
            char_embedding[self.charset_index[char]] = 1.  # Set character label
            if embed_with_class:
                char_embedding[self.num_lexical_features() + token_class] = 1.  # Set class label
            result.append(char_embedding)

        if corrupted_result is not None:
            for char, token_class in corrupted_char_class_seq:
                char_embedding = np.zeros(embedding_size)
                char_embedding[self.charset_index[char]] = 1.  # Set character label
                if embed_with_class:
                    char_embedding[self.num_lexical_features() + token_class] = 1.  # Set class label
                corrupted_result.append(char_embedding)

        # -- Append single eol vector
        assert len(result) < length_to_align
        char_embedding = np.zeros(embedding_size)
        char_embedding[self.charset_eol_index] = 1.
        if embed_with_class:
            char_embedding[self.num_lexical_features() + self.eol_class_id] = 1.
        result.append(char_embedding)
        if corrupted_result:
            char_embedding = np.zeros(embedding_size)
            if embed_with_class:
                char_embedding[self.charset_eol_index] = 1.
            corrupted_result.append(char_embedding)

        # -- Align output length by padding with null vecs
        true_sample_length = len(result)  # 2nd return value
        result += [np.zeros(embedding_size)] * (length_to_align - len(result))
        if corrupted_result:
            true_corrupted_sample_length = len(corrupted_result)
            corrupted_result += [np.zeros(embedding_size)] * \
                (length_to_align - len(corrupted_result) + corruption_grammar.max_corruptions())
        return (
            np.asarray(result, np.float32),
            true_sample_length,
            np.asarray(corrupted_result, np.float32),
            true_corrupted_sample_length)
