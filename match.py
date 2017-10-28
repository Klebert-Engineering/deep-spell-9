# (C) 2017 Klebert Engineering GmbH

import os
import sys
import pickle
import argparse
from scipy.spatial import cKDTree

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/models")

from deepspell.models.encoder import DSVariationalLstmAutoEncoder

arg_parser = argparse.ArgumentParser("NDS AutoCompletion Quality Evaluator")
arg_parser.add_argument(
    "--corpus",
    default="corpora/deepspell_data_north_america_cities.0",
    help="Path to the kdtree/token list from which correct matches should be drawn.")
arg_parser.add_argument(
    "--encoder",
    default="models/deepsp_spell-v1_na-lower_lr003_dec50_bat3072_emb8_fw128-128_bw128_de128-128_drop80.json",
    help="Path to the model JSON descriptor that should be used for token encoding.")
args = arg_parser.parse_args()

encoder_model = DSVariationalLstmAutoEncoder(args.encoder)
spell_tokens = [token.strip() for token in open(args.corpus+".tokens", "r")]
spell_kdtree = pickle.load(open(args.corpus+".kdtree", "rb"))

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
    lookup_vec = encoder_model.encode(user_command)
    print("Nearest to", lookup_vec, ":")
    _, query_result_indices = spell_kdtree.query(lookup_vec, k=3)
    for i in query_result_indices:
        print(spell_tokens[i], spell_kdtree.data[i])
