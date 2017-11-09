# (C) 2017 Klebert Engineering GmbH

import os
import sys
import pickle
import argparse
import math
import random
from collections import defaultdict
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/modules")

from deepspell.corpus import DSCorpus
from deepspell.grammar import DSGrammar
from deepspell.baseline.symspell import DSSymSpellBaseline
from deepspell.models.encoder import DSVariationalLstmAutoEncoder

arg_parser = argparse.ArgumentParser("NDS AutoCompletion Quality Evaluator")
arg_parser.add_argument(
    "--grammar",
    default="corpora/grammar-encoder.json",
    help="Path to the JSON descriptor for the grammar that should be used for sample gen.")
arg_parser.add_argument(
    "--corpus",
    default="corpora/deepspell_data_north_america_cities.tsv",
    help="Path to the kdtree/token list from which correct matches should be drawn.")
arg_parser.add_argument(
    "--model",
    default="models/deepsp_spell-v1_na-lower_lr003_dec50_bat3072_emb8_fw128-128_bw128_de128-128_drop80.json",
    help="Path to the model JSON descriptor that should be used for token encoding.")
arg_parser.add_argument(
    "--model-data",
    default="corpora/deepspell_data_north_america_cities.0",
    help="Path to the model JSON descriptor that should be used for token encoding.",
    dest="model_data")
arg_parser.add_argument(
    "--baseline",
    default=False,
    action="store_true",
    help="Use this flag in place of --model if you wish to use the baseline matcher.")
arg_parser.add_argument(
    "-s", "--test-split-percentage",
    default=100,
    type=int,
    dest="test_split",
    choices=range(100),
    help="Per-token-class percentage of the given corpus data that should be used for evaluation.")
args = arg_parser.parse_args()

print("Benchmarking FTS Spellchecker... ")
print("  ... model:      "+("SYMSPELL-BASELINE" if args.baseline else args.model))
print("  ... model_data: "+args.model_data)
print("  ... corpus:     "+args.corpus)
print("  ... grammar:    "+args.grammar)
print("=======================================================================")
print("")

if args.baseline:
    encoder_model = DSSymSpellBaseline(args.model_data)
    spell_tokens = None
    spell_kdtree = None
else:
    encoder_model = DSVariationalLstmAutoEncoder(args.model)
    spell_tokens = [token.strip() for token in open(args.model_data+".tokens", "r")]
    spell_kdtree = pickle.load(open(args.model_data+".kdtree", "rb"))

training_corpus = DSCorpus(args.corpus, "na", lowercase=True)
training_grammar = DSGrammar(args.grammar, training_corpus.featureset)
featureset = training_corpus.featureset


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


class TokenClassBenchmark:
    def __init__(self):
        self.corrected = 0
        self.not_corrected = 0

    def correction_precision(self):
        total, correct = self.corrected+self.not_corrected, self.corrected
        if total == 0:
            return .0
        return float(correct)/float(total)


def test_split_len(token_list):
    return math.ceil(len(token_list) * float(args.test_split)/100.)


# This dictionary contains a TokenClassBenchmark object for every
# token class in the given corpus.
benchmark_stats = defaultdict(lambda: TokenClassBenchmark())

# These variables serve to observe the evaluation progress
total_tokens = sum(test_split_len(tokens) for _, tokens in training_corpus.data.items())
completed_tokens = 0

for class_id, tokens in training_corpus.data.items():

    random.shuffle(tokens)

    for test_token in tokens[:test_split_len(tokens)]:

        completed_tokens += 1
        print_progress(completed_tokens, total_tokens, suffix="({}/{}) ('{}')".format(
            completed_tokens,
            total_tokens,
            test_token.string))
        test_string = training_grammar.random_phrase_with_token(test_token, corrupt=True)[0].string  # Get sample phrase

        # -- Evaluate corrector
        if args.baseline:
            suggestions = {suggestion[0] for suggestion in encoder_model.match(test_string, silent=True)[:3]}
        else:
            lookup_vector = encoder_model.encode(test_string)
            suggestions = {spell_tokens[i] for i in spell_kdtree.query(lookup_vector, k=3)[1]}

        if test_token.string in suggestions:
            benchmark_stats[class_id].corrected += 1
        else:
            benchmark_stats[class_id].not_corrected += 1

print("\n\nDone.")
print("Results:")
for class_id, benchmark in benchmark_stats.items():
    print(" * {}: corr={}%".format(
        featureset.class_name_for_id(class_id),
        benchmark.correction_precision()))
