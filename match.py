# (C) 2017 Klebert Engineering GmbH

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/modules")

from deepspell.baseline.symspell import DSSymSpellBaseline
from deepspell.models.encoder import DSVariationalLstmAutoEncoder
from deepspell.token_lookup_space import DSTokenLookupSpace

arg_parser = argparse.ArgumentParser("NDS AutoCompletion Quality Evaluator")
arg_parser.add_argument(
    "--corpus",
    default="corpora/deepspell_data_north_america_cities.1",
    help="Path to the kdtree/token list from which correct matches should be drawn.")
arg_parser.add_argument(
    "--encoder",
    default="models/deepsp_spell-v2_na-lower_lr003_dec70_bat2048_emb8_fw128_bw128_co256-256_dein256-256_drop75.json",
    help="Path to the model JSON descriptor that should be used for token encoding.")
arg_parser.add_argument(
    "--baseline",
    default=False,
    action="store_true",
    help="Use this flag in place of --encoder if you wish to use the baseline matcher.")
args = arg_parser.parse_args()


if args.baseline:
    spellchecker = DSSymSpellBaseline(args.corpus)
else:
    spellchecker = DSTokenLookupSpace(DSVariationalLstmAutoEncoder(args.encoder), args.corpus)

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
    if len(user_command) == 0:
        continue
    if user_command == "q":
        exit(0)
    print(spellchecker.match(user_command.lower()))
