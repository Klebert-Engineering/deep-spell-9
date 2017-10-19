# (C) 2017 Klebert Engineering GmbH

import sys
import os
import random
import math
import argparse
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.corpus import DSCorpus
from deepspell.grammar import DSGrammar
from deepspell.extrapolator import DSLstmExtrapolator
from deepspell.discriminator import DSLstmDiscriminator
from deepspell.baseline import DSFts5BaselineCompleter

arg_parser = argparse.ArgumentParser("NDS AutoCompletion Quality Evaluator")
arg_parser.add_argument(
    "--corpus",
    default="corpora/deepspell_data_north_america_v2.tsv",
    help="Path to the corpus from which benchmark samples should be drawn.")
arg_parser.add_argument(
    "--discr",
    default="models/deepsp_discr-v1_na_lr003_dec50_bat3072_fw128-128_bw128.json",
    help="Path to the model JSON descriptor that should be used for class discrimination.")
arg_parser.add_argument(
    "--no-discr",
    dest="skip_discriminator",
    default=False,
    action="store_true",
    help="Flag to indicate, whether the discriminator should be evaluated.")
arg_parser.add_argument(
    "--extra",
    default="models/deepsp_extra-v2_na_lr003_dec50_bat2048_256-256.json",
    # "models/deepsp_extra-v1_na_lr003_dec50_bat4096_128-128.json"
    # "models/deepsp_extra-v2_na_lr003_dec50_bat3072_128-128-128.json"
    help="Path to the model JSON descriptor that should be used for completion generation.")
arg_parser.add_argument(
    "--baseline",
    default=False,
    action="store_true",
    help="Use this flag in place of --extra if you wish to evaluate the baseline extrapolator.")
arg_parser.add_argument(
    "--grammar",
    default="corpora/grammar-address-na.json",
    help="Path to the JSON descriptor for the grammar that should be used for sample gen.")
arg_parser.add_argument(
    "-c", "--completions",
    default=3,
    type=int,
    help="Number of top suggestions that should be considered relevant for evaluation.")
arg_parser.add_argument(
    "-s", "--test-split-percentage",
    default=1,
    type=int,
    dest="test_split",
    choices=range(100),
    help="Per-token-class percentage of the given corpus data that should be used for evaluation.")
arg_parser.add_argument(
    "-p", "--prefix-sizes",
    default=(0, 1, 2, 3, 4, 5),
    nargs="+",
    type=int,
    dest="prefix_sizes",
    metavar="N",
    help="Position where a test-token should be cut off, and the remaining postfix evaluated against completion.")
args = arg_parser.parse_args()

print("Benchmarking FTS AutoCompleter... ")
print("  ... discriminator: "+args.discr)
print("  ... extrapolator:  "+("FTS-5-BASELINE" if args.baseline else args.extra))
print("  ... corpus:        "+args.corpus)
print("  ... grammar:       "+args.grammar)
print("=======================================================================")
print("")

training_corpus = DSCorpus(args.corpus, "na", lowercase=True)
training_grammar = DSGrammar(args.grammar, training_corpus.featureset)
featureset = training_corpus.featureset

if not args.skip_discriminator:
    discriminator_model = DSLstmDiscriminator(args.discr, "logs")
    # -- This is unfortunately necessary for some older pre-spellcheck models,
    #  which do not carry the BOL char in their charset.
    featureset.charset = discriminator_model.featureset.charset
    assert training_corpus.featureset.is_compatible(discriminator_model.featureset)

if args.baseline:
    extrapolator_model = DSFts5BaselineCompleter(training_corpus)
else:
    extrapolator_model = DSLstmExtrapolator(args.extra, "logs")
    assert extrapolator_model.featureset.is_compatible(discriminator_model.featureset)


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
        # `completion_accuracies` contains a mapping
        # from prefix length to counts of completed
        # / correctly completed occurences.
        self.completion_accuracies = defaultdict(lambda: [0, 0])
        self.identified = 0
        self.correctly_identified = 0
        self.incorrectly_identified = 0
        self.not_identified = 0

    def extrapolation_precision(self, prefix_length):
        total, correct = self.completion_accuracies[prefix_length]
        if total == 0:
            return .0
        return float(correct)/float(total)

    def discrimination_precision(self):
        if self.identified == 0:
            return .0
        return float(self.correctly_identified)/float(self.identified)

    def discrimination_recall(self):
        if self.identified == 0:
            return .0
        return float(self.correctly_identified)/float(self.correctly_identified+self.not_identified)

    def discrimination_f1(self):
        if self.identified == 0:
            return .0
        prec = self.discrimination_precision()
        rec = self.discrimination_recall()
        return (2.0*prec*rec)/(prec+rec)


def test_split_len(token_list):
    return math.ceil(len(token_list) * float(args.test_split)/100.)


def embed_truncated_token_sequence(token_sequence, truncate_token=None, prefix_length=-1):
    if (prefix_length >= 0) and (prefix_length < len(truncate_token.string)):
        remaining_postfix = truncate_token.string[prefix_length:]
    else:
        remaining_postfix = ""
    embedded_chars = ""
    embedded_classes = []
    for i, token in enumerate(token_sequence):
        token_str = (" " if i > 0 else "") + token.string
        if token == truncate_token:
            token_str = token_str[:prefix_length + (1 if i > 0 else 0)]
        embedded_chars += token_str
        embedded_classes += [token.id[0]] * len(token_str)
        if token == truncate_token:
            break
    return embedded_chars, embedded_classes, remaining_postfix


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
        test_phrase = training_grammar.random_phrase_with_token(test_token)  # Get sample phrase

        # -- Evaluate discriminator
        if not args.skip_discriminator:
            phrase_chars, phrase_classes, gold_completion = embed_truncated_token_sequence(test_phrase)
            class_labels = discriminator_model.discriminate(featureset, phrase_chars)
            class_labels = [featureset.class_ids[class_pd[0][0]] for class_pd in class_labels][:-1]
            assert len(phrase_classes) == len(class_labels)
            for gold_class, labeled_class in zip(phrase_classes, class_labels):
                if gold_class == labeled_class:
                    benchmark_stats[gold_class].identified += 1
                    benchmark_stats[gold_class].correctly_identified += 1
                else:
                    benchmark_stats[gold_class].identified += 1
                    benchmark_stats[gold_class].not_identified += 1
                    benchmark_stats[labeled_class].identified += 1
                    benchmark_stats[labeled_class].incorrectly_identified += 1

        # -- Evaluate extrapolator
        for prefix_size in args.prefix_sizes:
            phrase_chars, phrase_classes, gold_completion = embed_truncated_token_sequence(
                test_phrase, test_token, prefix_size)
            if len(phrase_chars) == 0 or len(gold_completion) == 0:
                continue
            completions = extrapolator_model.extrapolate(featureset, phrase_chars, phrase_classes, len(gold_completion))
            completion_stats = benchmark_stats[test_token.id[0]].completion_accuracies[prefix_size]
            completion_stats[0] += len(gold_completion)
            best_completion_score = 0
            for completion, _, _ in completions[:args.completions]:
                # print(completion, "completes", phrase_chars)
                completion_score = 0
                for gold_char, label_char in zip(gold_completion, completion):
                    if gold_char == label_char:
                        completion_score += 1
                best_completion_score = max(best_completion_score, completion_score)
            completion_stats[1] += best_completion_score

print("\n\nDone.")
print("Results:")
for class_id, benchmark in benchmark_stats.items():
    print(" * {}: {}, discr/pr={}%, discr/re={}%, discr/f1={}%".format(
        featureset.class_name_for_id(class_id),
        ", ".join([
            "completion/prelen{}={}%".format(n, benchmark.extrapolation_precision(n) * 100)
            for n in args.prefix_sizes]),
        benchmark.discrimination_precision() * 100,
        benchmark.discrimination_recall() * 100,
        benchmark.discrimination_f1() * 100
    ))
