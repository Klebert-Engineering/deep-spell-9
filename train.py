
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.grammar import DSGrammar
from deepspell.corpus import DSCorpus
from deepspell.predictor import DSLstmPredictor

# training_corpus = DSCorpus("corpora/deepspell_minimal.tsv", "na")
training_corpus = DSCorpus("corpora/deepspell_data_north_america_v2.tsv", "na")
training_grammar = DSGrammar("corpora/grammar.json", training_corpus)
model = DSLstmPredictor(
    training_corpus.total_num_features_per_character(), "models", "logs",
    learning_rate=0.003,
    learning_rate_decay=0.9,
    training_epochs=20,
    batch_size=32,
    state_size_per_layer=(128, 128))

model.train(training_corpus, training_grammar)
