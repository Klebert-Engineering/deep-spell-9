
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.corpus import FtsCorpus, FtsGrammar
# from deepspell.predictor import FtsPredictor

corpus = FtsCorpus("corpora/deepspell_minimal.tsv")  # deepspell_data_north_america_v2.tsv
grammar = FtsGrammar("corpora/grammar.json", corpus)

corpus.get_batch_and_lengths(16, grammar)