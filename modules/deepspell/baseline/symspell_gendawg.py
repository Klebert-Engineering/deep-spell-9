# (C) 2018-present Klebert Engineering

"""
Opens a TSV FTS corpus file and generates misspelled entries
for each FTS token with a given maximum edit distance.
Takes two arguments:
(1) The corpus file
(2) The output file. Two output files will be generated from this argument:
    -> <output file>.refs : Contains pickled DAWG with misspelled tokens and correct token references
    -> <output file>.tokens : Contains the correctly spelled tokens, where line-number=reference-index,
       in the form of <token> <frequency>
"""

import codecs
import sys
import string
from hat_trie import Trie
from dawg import BytesDAWG


def generate_lookup_entries(w, max_edit_distance=0):
    """given a word, derive strings with up to max_edit_distance characters
       deleted"""
    result = {w}
    queue = {w}
    for d in range(max_edit_distance):
        temp_queue = set()
        for word in queue:
            if len(word) > 1:
                for c in range(len(word)):  # character index
                    word_minus_c = word[:c] + word[c + 1:]
                    if word_minus_c not in result:
                        result.add(word_minus_c)
                    if word_minus_c not in temp_queue:
                        temp_queue.add(word_minus_c)
        queue = temp_queue
    return result


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=10):
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

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

bytes_for_token = Trie()  # charset
token_and_freq_for_index = []
longest_word_length = 0
bytes_per_index = 3

# Count total lines in corpus file
with codecs.open(input_file_path, encoding="utf-8") as corpus_file:
    total = sum(1 for _ in corpus_file)
done = 0

print("Loading completion tokens from '{}'...".format(input_file_path))
with codecs.open(input_file_path, encoding="utf-8") as input_file:
    index_for_token = Trie()  # charset
    for line in input_file:
        parts = line.split("\t")
        done += 1
        print_progress(done, total)
        if len(parts) < 6:
            continue
        token = parts[2].lower()  # unidecode.unidecode()
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections, frequency of word in corpus)
        if token in index_for_token:
            token_index = index_for_token[token]
        else:
            token_index = len(token_and_freq_for_index)
            index_for_token[token] = token_index
            longest_word_length = max(len(token), longest_word_length)
            token_and_freq_for_index.append([token, 0])
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word, but
            # counter of frequency of word in corpus is not incremented in those cases.
            deletes = generate_lookup_entries(token)
            word_index_bytes = token_index.to_bytes(bytes_per_index, 'big')
            for entry in deletes:
                if entry in bytes_for_token:
                    bytes_for_token[entry] += word_index_bytes
                else:
                    bytes_for_token[entry] = word_index_bytes

        # increment count of token in corpus
        token_and_freq_for_index[token_index][1] += 1
print("\n  ...done.")

print("Creating DAWG...")
dawg_dict = BytesDAWG(([token, bytes_for_token[token]] for token in bytes_for_token.iterkeys()))
print("  ...done.")

print("Writing output files {}.refs and {}.tokens ...".format(output_file_path, output_file_path))
dawg_dict.save(output_file_path + ".refs")
with codecs.open(output_file_path + ".tokens", "w", encoding="utf-8") as output_tokens:
    for token, freq in token_and_freq_for_index:
        output_tokens.write("{}\t{}\n".format(token, freq))
print("  ...done.")

