# (C) 2018-present Klebert Engineering

import codecs
from dawg import BytesDAWG


class DSSymSpellBaseline:

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, dawg_and_token_freq_file_path, max_edit_distance=2, bytes_per_index=3):
        """
        :param dawg_and_token_freq_file_path: Should point to a .tokens and .refs file.
        :param max_edit_distance: The maximum edit distance to be anticipated.
        """
        self.max_edit_distance = max_edit_distance
        self.longest_word_length = 0
        self.bytes_per_index = bytes_per_index

        print("Loading token frequencies from '{}.tokens'...".format(dawg_and_token_freq_file_path))
        self.token_and_freq_for_index = []
        with codecs.open(dawg_and_token_freq_file_path+".tokens") as token_file:
            for line in token_file:
                token, freq = line.strip().split("\t")
                self.longest_word_length = max(self.longest_word_length, len(token))
                self.token_and_freq_for_index.append((token, int(freq)))
        print("  ...done.")

        print("Loading spelling-DAWG from '{}.refs'...".format(dawg_and_token_freq_file_path))
        self.dictionary = BytesDAWG()
        self.dictionary.load(dawg_and_token_freq_file_path+".refs")
        print("  ... done.")

    def match(self, string, k=3, silent=True):
        """
        Obtain a list of @k corrections for the given @string.
        :param string: The token string which should be looked up, and whose
         most closely spelled correct tokens should be retrieved.
        :param k: The number of correction suggestions that should be retrieved.
        :param silent: Flag that determines whether any debug info is printed.
        :return: An ascendingly ordered list of @k (token, distance) pairs.
        """
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((k < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                reference_index_sequence_blob = self.dictionary[q_item][0]
                assert (len(reference_index_sequence_blob) % self.bytes_per_index) == 0

                for i in range(0, len(reference_index_sequence_blob), self.bytes_per_index):
                    sc_item_index = int.from_bytes(reference_index_sequence_blob[i:i+self.bytes_per_index], 'big')
                    assert sc_item_index < len(self.token_and_freq_for_index)
                    sc_item, sc_item_freq = self.token_and_freq_for_index[sc_item_index]
                    if sc_item not in suggest_dict:

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) >= len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = self._dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (k < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (sc_item_freq, item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if k < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (k < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and k != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # sort results by ascending order of edit distance and descending
        # return list of suggestions with (correction, edit distance)
        result_list = [(term_freq_dist_tuple[0], term_freq_dist_tuple[1][0])
            for term_freq_dist_tuple in sorted(
                suggest_dict.items(),
                key=lambda term_freq_dist_tuple: (term_freq_dist_tuple[1][1], -term_freq_dist_tuple[1][0])
            )[:k]]

        return result_list

    def best_word(self, s, silent=False):
        try:
            return self.match(s, silent)[0]
        except:
            return None

    # ----------------------[ Private Methods ]----------------------

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

