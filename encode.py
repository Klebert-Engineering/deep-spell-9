# (C) 2018-present Klebert Engineering

import argparse
import os
import sys
import pickle
import numpy as np
from scipy.spatial import cKDTree

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

# Load the corpus using DSCorpus
corpus = DSCorpus(args.corpus, "eu", lowercase=True)
encoder_model = DSVariationalLstmAutoEncoder(args.encoder, "logs")

# Process all tokens from the corpus
all_tokens = []
for class_tokens in corpus.data.values():
    all_tokens.extend(token.string for token in class_tokens)

# Encode tokens and build KD-tree
print(f"Encoding {len(all_tokens)} tokens...")
token_embeddings = np.array([encoder_model.encode(token) for token in all_tokens])
print("Building KD-tree...")
kdtree = cKDTree(token_embeddings)

# Save the tokens and kdtree
output_base = os.path.join(args.output_path, os.path.splitext(os.path.basename(args.corpus))[0])
print(f"Saving output files to {output_base}.tokens and {output_base}.kdtree ...")

with open(output_base + ".tokens", "w") as f:
    for token in all_tokens:
        f.write(token + "\n")

with open(output_base + ".kdtree", "wb") as f:
    pickle.dump(kdtree, f)

print("  ... done.")
