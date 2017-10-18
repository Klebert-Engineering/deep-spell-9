# (C) 2017 Klebert Engineering GmbH

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.grammar import DSGrammar
from deepspell.corpus import DSCorpus
from deepspell.encoder import DSVariationalLstmAutoEncoder

# training_corpus = DSCorpus("corpora/deepspell_minimal.tsv", "min", lowercase=True)
training_corpus = DSCorpus("corpora/deepspell_data_north_america_nozip_v2.tsv", "na", lowercase=True)
training_grammar = DSGrammar("corpora/grammar-encoder.json", training_corpus.featureset)
model = DSVariationalLstmAutoEncoder(
    "models", "logs",
    features=training_corpus.featureset,
    learning_rate=0.003,
    learning_rate_decay=0.5,
    training_epochs=5,
    batch_size=2048,
    encoder_fw_state_size_per_layer=[256],
    encoder_bw_state_size_per_layer=[256],
    decoder_state_size_per_layer=[256, 256],
    embedding_size=8,
    decoder_input_keep_prob=.9,
    kl_rate_rise_iterations=2000,
    kl_rate_rise_threshold=1000
)

model.train(training_corpus, training_grammar)
