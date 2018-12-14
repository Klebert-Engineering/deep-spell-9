# (C) 2018-present Klebert Engineering

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.corpus import DSCorpus
from deepspell.grammar import DSGrammar
from deepspell_optimization.models.encoder import DSVariationalLstmAutoEncoderOptimizer

# training_corpus = DSCorpus("corpora/deepspell_minimal.tsv", "min", lowercase=True)
training_corpus = DSCorpus("corpora/deepspell_data_north_america_nozip_v2.tsv", "na", lowercase=True)
training_grammar = DSGrammar("corpora/grammar-encoder.json", training_corpus.featureset)
model = DSVariationalLstmAutoEncoderOptimizer(
    "models", "logs",
    features=training_corpus.featureset,
    learning_rate=0.003,
    learning_rate_decay=0.7,
    training_epochs=10,
    batch_size=2048,
    encoder_fw_state_size_per_layer=[128],
    encoder_bw_state_size_per_layer=[128],
    encoder_combine_state_size_per_layer=[256, 256],
    decoder_state_size_per_layer=[256, 256],
    embedding_size=8,
    latent_space_as_decoder_state=False,
    decoder_input_keep_prob=.25,
    kl_rate_rise_iterations=1,
    kl_rate_rise_threshold=1000000
)

model.train(training_corpus, training_grammar)
