# (C) 2018-present Klebert Engineering

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.corpus import DSCorpus
from deepspell.grammar import DSGrammar
from deepspell_optimization.models.discriminator import DSLstmDiscriminatorOptimizer

# training_corpus = DSCorpus("corpora/deepspell_minimal.tsv", "na-min")
# training_corpus = DSCorpus("corpora/deepspell_data_north_america_nozip_v2.tsv", "na", lowercase=True)
training_corpus = DSCorpus("corpora/RoadFTS5_EU.NDS", "eu", lowercase=True)
training_grammar = DSGrammar("corpora/grammar-address-na.json", training_corpus.featureset)
model = DSLstmDiscriminatorOptimizer(
    "models", "logs",
    features=training_corpus.featureset,
    learning_rate=0.003,
    learning_rate_decay=0.5,
    training_epochs=5,
    batch_size=4096,
    fw_state_size_per_layer=(128, 256),
    bw_state_size_per_layer=(128, ),
    min_sample_length_before_truncation=3)

model.train(training_corpus, training_grammar)
