# (C) 2017 Klebert Engineering GmbH

import sys
import os
import random
import math
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.corpus import DSCorpus
from deepspell.grammar import DSGrammar
from deepspell.extrapolator import DSLstmExtrapolator
from deepspell.discriminator import DSLstmDiscriminator

corpus_path = "corpora/deepspell_data_north_america_v2.tsv"
discr_path = "models/deepsp_discr-v1_na_lr003_dec50_bat3072_fw128-128_bw128.json"
extra_path = "models/deepsp_extra-v2_na_lr003_dec50_bat2048_256-256.json"
# "models/deepsp_extra-v1_na_lr003_dec50_bat4096_128-128.json"
# "models/deepsp_extra-v2_na_lr003_dec50_bat3072_128-128-128.json"

training_corpus = DSCorpus(corpus_path, "na")
extrapolator_model = DSLstmExtrapolator(extra_path, "logs")
discriminator_model = DSLstmDiscriminator(discr_path, "logs")
assert extrapolator_model.featureset.is_compatible(discriminator_model.featureset)
assert training_corpus.featureset.is_compatible(discriminator_model.featureset)
featureset = extrapolator_model.featureset
training_grammar = DSGrammar("corpora/grammar.json", training_corpus.featureset)

print("Benchmarking FTS AutoCompleter... ")
print("  ... discriminator: "+discr_path)
print("  ... extrapolator:  "+extra_path)
print("  ... corpus:        "+corpus_path)
print("  ... grammar:       "+training_grammar.root_nonterminal)
print("=======================================================================")
print("")

MIN_CUTOFF_POINT = 2
MAX_COMPLETE_LENGTH = 8


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
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
        self.completed = 0
        self.correctly_completed = 0
        self.identified = 0
        self.correctly_identified = 0
        self.incorrectly_identified = 0
        self.not_identified = 0

    def greedy_extrapolation_precision(self):
        return float(self.correctly_completed)/float(self.completed)

    def discrimination_precision(self):
        return float(self.correctly_identified)/float(self.identified)

    def discrimination_recall(self):
        return float(self.correctly_identified)/float(self.correctly_identified+self.not_identified)

    def discrimination_f1(self):
        prec = self.discrimination_precision()
        rec = self.discrimination_recall()
        return (2.0*prec*rec)/(prec+rec)

benchmark_stats = defaultdict(lambda: TokenClassBenchmark())

total_tokens = sum(math.ceil(len(tokens)*.005) for _, tokens in training_corpus.data.items())
completed_tokens = 0

for class_id, tokens in training_corpus.data.items():

    for token in tokens[:math.ceil(len(tokens)*.005)]:

        completed_tokens += 1
        print_progress(completed_tokens, total_tokens)

        phrase_tokens = training_grammar.random_phrase_with_token(token)  # Get sample phrase
        phrase_tokens = phrase_tokens[:random.choice(range(len(phrase_tokens)))+1]  # Randomly chop off
        while len(phrase_tokens[-1].string)-1 <= MIN_CUTOFF_POINT:
            phrase_tokens = phrase_tokens[:-1]
            if not phrase_tokens:
                break
        if not phrase_tokens:
            continue

        token_to_complete = phrase_tokens[-1]
        token_to_complete_cutoff_point = math.ceil(random.uniform(MIN_CUTOFF_POINT, len(token_to_complete.string)-2))
        if token_to_complete_cutoff_point >= len(token_to_complete.string):
            token_to_complete_cutoff_point = len(token_to_complete.string)-2
        token_to_complete_postfix = token_to_complete.string[token_to_complete_cutoff_point:]

        phrase_chars = ""
        phrase_classes = []

        for i, phrase_token in enumerate(phrase_tokens):

            token_str = (" " if i > 0 else "") + phrase_token.string
            if phrase_token == token_to_complete:
                token_str = token_str[:token_to_complete_cutoff_point+(1 if i > 0 else 0)]

            phrase_chars += token_str
            phrase_classes += [phrase_token.id[0]] * len(token_str)

        # -- Evaluate discriminator
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
        completion = extrapolator_model.extrapolate(featureset, phrase_chars, phrase_classes, len(token_to_complete_postfix))
        completion = [char_pd[0][0] for char_pd in completion[0]]

        benchmark_stats[token_to_complete.id[0]].completed += len(token_to_complete_postfix)
        for gold_char, label_char in zip(token_to_complete_postfix, completion):
            if gold_char == label_char:
                benchmark_stats[token_to_complete.id[0]].correctly_completed += 1

print("\n\nDone.")
print("Results:")
for class_id, benchmark in benchmark_stats.items():
    print(" * {}: extra_prec: {}%, discr_prec: {}%, discr_recall: {}%, discr_f1: {}%".format(
        featureset.class_name_for_id(class_id),
        benchmark.greedy_extrapolation_precision()*100,
        benchmark.discrimination_precision()*100,
        benchmark.discrimination_recall()*100,
        benchmark.discrimination_f1()*100
    ))
