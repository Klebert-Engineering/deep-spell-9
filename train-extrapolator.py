# (C) 2017 Klebert Engineering GmbH

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.corpus import DSCorpus
from deepspell_optimization.grammar import DSGrammar
from deepspell_optimization.models.extrapolator import DSLstmExtrapolatorOptimizer

# training_corpus = DSCorpus("corpora/deepspell_minimal.tsv", "na")
training_corpus = DSCorpus("corpora/deepspell_data_north_america_nozip_v2.tsv", "na", lowercase=True)
training_grammar = DSGrammar("corpora/grammar-address-na.json", training_corpus.featureset)
model = DSLstmExtrapolatorOptimizer(
    "models", "logs",
    features=training_corpus.featureset,
    learning_rate=0.003,
    learning_rate_decay=0.5,
    training_epochs=5,
    batch_size=3072,
    state_size_per_layer=(256, 256))

model.train(training_corpus, training_grammar)
