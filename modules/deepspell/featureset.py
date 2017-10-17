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

    LOWER_CASE_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789-., /_$"
    FULL_CASE_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + LOWER_CASE_CHARSET

    def __init__(self,
                 classes=None,
                 charset=LOWER_CASE_CHARSET,
                 eol_char="$",
                 unk_char="_",
                 eol_class_name="EOL"):

        assert not classes or isinstance(classes, dict)
        self.class_ids = classes
        self.charset = charset
        self.charset_unk_index = charset.index(unk_char)
        self.charset_eol_index = charset.index(eol_char)
        self.charset_index = defaultdict(lambda: self.charset_unk_index, ((c, i) for i, c in enumerate(charset)))
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

    def embed_tokens(self, token_list, length_to_align, min_chars_truncate):
        """
        Embeds a sequence of FtsToken instances into a 2D feature matrix
        like [num_characters][total_num_features_per_character()].
        :return: 2D Embedding feature matrix and length of sample
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
            char_embedding[self.charset_index[char]] = 1.
            # Set class label
            char_embedding[self.num_lexical_features() + token_class] = 1.
            result.append(char_embedding)

        # -- Append single eol vector
        assert len(result) < length_to_align
        char_embedding = np.zeros(self.total_num_features_per_character())
        char_embedding[self.charset_eol_index] = 1.
        char_embedding[self.num_lexical_features() + self.eol_class_id] = 1.
        result.append(char_embedding)

        # -- Align output length by padding with null vecs
        true_sample_length = len(result)
        while len(result) < length_to_align:
            result.append(np.zeros(self.total_num_features_per_character()))
        return np.asarray(result, np.float32), true_sample_length
