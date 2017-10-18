# (C) 2017 Klebert Engineering GmbH

import sys
import os
import random
import math
import argparse
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.corpus import DSCorpus
from deepspell.encoder import DSVariationalLstmAutoEncoder

arg_parser = argparse.ArgumentParser("NDS AutoCompletion Quality Evaluator")
arg_parser.add_argument(
    "--corpus",
    default="corpora/deepspell_data_north_america_v2.tsv",
    help="Path to the corpus from which benchmark samples should be drawn.")
arg_parser.add_argument(
    "--encoder",
    default="models/deepsp_spell-v1_na_lr003_dec50_bat3072_fw128-128_bw128.json",
    help="Path to the model JSON descriptor that should be used for token encoding.")
arg_parser.add_argument(
    "--output-dir", "-o",
    dest="output_path",
    default="corpora/encoded.dict",
    help="Directory path to where the generated embeddings should be stored.")
args = arg_parser.parse_args()

print("Encoding FTS Corpus... ")
print("  ... encoder:  "+args.encoder)
print("  ... corpus:   "+args.corpus)
print("=======================================================================")
print("")

corpus_to_encode = DSCorpus(args.corpus, "na", lowercase=True)
encoder_model = DSVariationalLstmAutoEncoder(args.encoder, "logs")
assert corpus_to_encode.featureset.is_compatible(encoder_model.featureset)
featureset = encoder_model.featureset

encoder_model.encode(corpus_to_encode, 16384, args.output_path)
