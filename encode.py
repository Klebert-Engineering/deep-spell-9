# (C) 2017 Klebert Engineering GmbH

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.corpus import DSCorpus
from deepspell.models.encoder import DSVariationalLstmAutoEncoder

arg_parser = argparse.ArgumentParser("NDS AutoCompletion Quality Evaluator")
arg_parser.add_argument(
    "--corpus",
    # default="corpora/deepspell_data_north_america_nozip_v2.tsv",
    default="corpora/deepspell_minimal.tsv",
    help="Path to the corpus from which benchmark samples should be drawn.")
arg_parser.add_argument(
    "--encoder",
    default="models/deepsp_spell-v1_na-lower_lr003_dec50_bat3072_emb8_fw128-128_bw128_de128-128_drop80.json",
    help="Path to the model JSON descriptor that should be used for token encoding.")
arg_parser.add_argument(
    "--output-dir", "-o",
    dest="output_path",
    default="corpora/",
    help="Directory path to where the generated embeddings should be stored.")
arg_parser.add_argument(
    "--batch-size", "-b",
    dest="batch_size",
    default=16384,
    type=int,
    help="Number of samples that should be processed in parallel.")
args = arg_parser.parse_args()

print("Encoding FTS Corpus... ")
print("  ... encoder:  "+args.encoder)
print("  ... corpus:   "+args.corpus)
print("=======================================================================")
print("")

encoder_model = DSVariationalLstmAutoEncoder(args.encoder, "logs")
encoder_model.encode_corpus(args.corpus, args.output_path, batch_size=args.batch_size)
