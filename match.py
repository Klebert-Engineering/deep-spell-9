# (C) 2017 Klebert Engineering GmbH

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/modules")

from deepspell.corpus import DSEncodedCorpus
from deepspell.encoder import DSVariationalLstmAutoEncoder

encoder_model = DSVariationalLstmAutoEncoder("models/deepsp_spell-v1_na-lower_lr003_dec50_bat3072_emb8_fw128-128_bw128_de128-128_drop80.json")
match_corpus = DSEncodedCorpus("corpora/na-lower.7.vectors.bin")
featureset = encoder_model.featureset

print("""
=============================================================
  This is the `deepspell` LSTM DNN spell-match demo for NDS
             (C) 2017 Klebert Engineering GmbH
=============================================================

Usage:
 -> Enter a string that should be encoded/matched.

Have fun!

""")

while True:
    user_command = input("> ")
    if user_command == "q":
        exit(0)
    lookup_vec = encoder_model.encode(user_command)
    print(match_corpus.lookup(lookup_vec, n=3))
