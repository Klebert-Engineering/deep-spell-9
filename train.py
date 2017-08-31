# (C) 2017 Klebert Engineering GmbH

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
    "models", "logs",
    num_lexical_features=training_corpus.total_num_lexical_features_per_character(),
    num_logical_features=training_corpus.total_num_logical_features_per_character(),
    learning_rate=0.003,
    learning_rate_decay=0.5,
    training_epochs=10,
    batch_size=4096,
    extrapolator_state_size_per_layer=(256, 256),
    discriminator_forward_state_size_per_layer=(256,),
    discriminator_backward_state_size_per_layer=(256,))

model.train_extrapolator(training_corpus, training_grammar)
