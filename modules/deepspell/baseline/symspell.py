"""
symspell_python.py

################

To run, execute python symspell_python.py at the prompt.
Make sure the dictionary "big.txt" is in the current working directory.
Enter word to correct when prompted.

################

v 1.3 last revised 29 Apr 2017
Please note: This code is no longer being actively maintained.

License:
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License,
version 3.0 (LGPL-3.0) as published by the Free Software Foundation.
http://www.opensource.org/licenses/LGPL-3.0

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

Please acknowledge Wolf Garbe, as the original creator of SymSpell,
(see note below) in any use.

################

This program is a Python version of a spellchecker based on SymSpell,
a Symmetric Delete spelling correction algorithm developed by Wolf Garbe
and originally written in C#.

From the original SymSpell documentation:

"The Symmetric Delete spelling correction algorithm reduces the complexity
 of edit candidate generation and dictionary lookup for a given Damerau-
 Levenshtein distance. It is six orders of magnitude faster and language
 independent. Opposite to other algorithms only deletes are required,
 no transposes + replaces + inserts. Transposes + replaces + inserts of the
 input term are transformed into deletes of the dictionary term.
 Replaces and inserts are expensive and language dependent:
 e.g. Chinese has 70,000 Unicode Han characters!"

For further information on SymSpell, please consult the original
documentation:
  URL: blog.faroo.com/2012/06/07/improved-edit-distance-based-spelling-correction/
  Description: blog.faroo.com/2012/06/07/improved-edit-distance-based-spelling-correction/

The current version of this program will output all possible suggestions for
corrections up to an edit distance (configurable) of max_edit_distance = 3.

With the exception of the use of a third-party method for calculating
Demerau-Levenshtein distance between two strings, we have largely followed
the structure and spirit of the original SymSpell algorithm and have not
introduced any major optimizations or improvements.

################

Changes from version (1.0):
We implement allowing for less verbose options: e.g. when only a single
recommended correction is required, the search may terminate early, thereby
enhancing performance.

Changes from version (1.1):
Removed unnecessary condition in create_dictionary_entry

Changes from version (1.2):
Update maintenance status

#################

Sample output:

Please wait...
Creating dictionary...
total words processed: 1105285
total unique words in corpus: 29157
total items in dictionary (corpus words and deletions): 2151998
  edit distance for deletions: 3
  length of longest word in corpus: 18

Word correction
---------------
Enter your input (or enter to exit): there
('there', (2972, 0))

Enter your input (or enter to exit): hellot
('hello', (1, 1))

Enter your input (or enter to exit): accomodation
('accommodation', (5, 1))

Enter your input (or enter to exit):
goodbye
"""

import time
import pygtrie
import sys
import codecs

from dawg import BytesDAWG
from collections import defaultdict


class DSSymSpellBaseline:

    # ---------------------[ Interface Methods ]---------------------

    def __init__(self, completion_corpus, max_edit_distance=2, verbose=1, bytes_per_index=3):
        """
        :param completion_corpus: Tab-separated FTS corpus file path with tokens to read.
        :param max_edit_distance: The maximum edit distance to be anticipated.
        :param verbose: Determines, how many values should be retrieved with each match() call.
        0: top suggestion
        1: all suggestions of smallest edit distance
        2: all suggestions <= max_edit_distance (slower, no early termination)
        """
        self.verbose = verbose
        self.max_edit_distance = max_edit_distance
        self.longest_word_length = 0
        self.bytes_per_index = bytes_per_index
        print("Loading completion tokens from '{}'...".format(completion_corpus))
        with codecs.open(completion_corpus) as corpus_file:
            total = sum(1 for _ in corpus_file)
        done = 0
        index_for_token = dict()
        bytes_for_token = defaultdict(lambda: b"")
        self.token_and_freq_for_index = []

        with codecs.open(completion_corpus) as corpus_file:
            for line in corpus_file:
                parts = line.split("\t")
                done += 1
                self._print_progress(done, total)
                if len(parts) < 6:
                    continue
                token = parts[2].lower()
                # check if word is already in dictionary
                # dictionary entries are in the form: (list of suggested corrections, frequency of word in corpus)
                if token in index_for_token:
                    token_index = index_for_token[token]
                else:
                    token_index = len(self.token_and_freq_for_index)
                    index_for_token[token] = token_index
                    self.longest_word_length = max(len(token), self.longest_word_length)
                    self.token_and_freq_for_index.append([token, 0])
                    # first appearance of word in corpus
                    # n.b. word may already be in dictionary as a derived word, but
                    # counter of frequency of word in corpus is not incremented in those cases.
                    deletes = self._generate_lookup_entries(token)
                    word_index_bytes = token_index.to_bytes(self.bytes_per_index, 'big')
                    for entry in deletes:
                        bytes_for_token[entry] += word_index_bytes

                # increment count of token in corpus
                self.token_and_freq_for_index[token_index][1] += 1

        self.dictionary = BytesDAWG(bytes_for_token.items())
        print("\n  ... done.")

    def match(self, string, silent=False):
        """
        Returns list of suggested corrections for potentially incorrectly spelled word.

        Option 1:
        ['file', 'five', 'fire', 'fine', ...]

        Option 2:
        [('file', (5, 0)),
         ('five', (67, 1)),
         ('fire', (54, 1)),
         ('fine', (17, 1))...]
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
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
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

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        item_dist = self._dameraulevenshtein(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        if (self.verbose < 2) and (item_dist > min_suggest_len):
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
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction, (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        result_list = sorted(
            as_list,
            key=lambda term_freq_dist_tuple: (term_freq_dist_tuple[1][1], -term_freq_dist_tuple[1][0]))

        if self.verbose == 0:
            return result_list[0]
        else:
            return result_list

    def best_word(self, s, silent=False):
        try:
            return self.match(s, silent)[0]
        except:
            return None

    # ----------------------[ Private Methods ]----------------------

    def _generate_lookup_entries(self, w):
        """given a word, derive strings with up to max_edit_distance characters
           deleted"""
        deletes = [w]
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue
        return deletes

    def _add_word(self, word):
        """add word and its derived deletions to dictionary"""
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections, frequency of word in corpus)
        new_real_word_added = False
        if word in self.dictionary:
            # increment count of word in corpus
            self.dictionary[word] = (self.dictionary[word][0], self.dictionary[word][1] + 1)
        else:
            self.longest_word_length = max(len(word), self.longest_word_length)
            self.dictionary[word] = ([], 1)

        if self.dictionary[word][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word, but
            # counter of frequency of word in corpus is not incremented in those cases.
            new_real_word_added = True
            deletes = self._generate_lookup_entries(word)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    self.dictionary[item][0].append(word)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = ([word], 0)
        return new_real_word_added

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

    @staticmethod
    def _print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=10):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            bar_length  - Optional  : character length of bar (Int)
        """
        str_format = "{0:." + str(decimals) + "f}"
        percents = str_format.format(100 * (iteration / float(total)))
        filled_length = int(round(bar_length * iteration / float(total)))
        bar = '#' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
        if iteration == total:
            sys.stdout.write('\n')
        sys.stdout.flush()
