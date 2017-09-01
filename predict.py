# (C) 2017 Klebert Engineering GmbH

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.grammar import DSGrammar
from deepspell.corpus import DSCorpus
from deepspell.extrapolator import DSLstmExtrapolator

# training_corpus = DSCorpus("corpora/deepspell_minimal.tsv", "na")
training_corpus = DSCorpus("corpora/deepspell_data_north_america_v2.tsv", "na")
training_grammar = DSGrammar("corpora/grammar.json", training_corpus)
model = DSLstmExtrapolator("models/deepspell_lstm_v1_na_lr003_dec70_bat4096.json", "logs")

print("""
=============================================================
This is the `deepspell` LSTM DNN auto-completion demo for NDS
             (C) 2017 Klebert Engineering GmbH
=============================================================

Usage:
 -> Enter a prefix to complete, or type 'q' to quit.
 -> A prefix consists of two parts: a character-sequence and
    a class-sequence. The following class ids are available:
{}
 -> To obtain a prefix completion, type the prefix character
    sequence followed by '~', and then the numeric class-id-
    sequence for each character in the prefix:
    
    e.g.
           Los Angeles Calif~00000000000111111
      
Have fun!

""".format(
    "\n".join("    {}: {}".format(class_name, class_id) for class_name, class_id in training_corpus.class_ids.items())
))

while True:
    user_command = input("> ")
    if user_command == "q":
        exit(0)

    parts = user_command.split("~")
    if len(parts) < 2:
        print("Bad input!")
        continue

    prefix_chars, prefix_classes = parts
    prefix_classes = prefix_classes.strip()
    if len(prefix_chars) != len(prefix_classes):
        print("The char and class inputs are not of equal length!")
        continue

    prefix_class_names = []
    for cl in prefix_classes:
        cl_name = training_corpus.class_name_for_id(int(cl))
        if cl_name:
            prefix_class_names.append(cl_name)
        else:
            print("{} is not a valid class id!".format(cl))
            prefix_class_names = []
            break

    if not prefix_class_names:
        continue

    completion_chars, completion_classes = model.extrapolate(training_corpus, prefix_chars, prefix_class_names, 24)
    char_cols = [[] for _ in range(len(completion_chars))]
    class_cols = [[] for _ in range(len(completion_chars))]
    for t in range(len(completion_chars)):
        for i in range(3):
            char_cols[t].append(" {} {}% ".format(completion_chars[t][i][0], str(completion_chars[t][i][1])[2:4]))
        for i in range(3):
            class_cols[t].append(" {} {}% ".format(completion_classes[t][i][0][:2], str(completion_classes[t][i][1])[2:4]))
    max_col_width = max(len(s) for col in class_cols+char_cols for s in col)
    for line in range(len(char_cols[0])):
        print(" " + "|".join(col[line].ljust(max_col_width) for col in char_cols))
    print(" " + "|".join(["-"*max_col_width] * len(completion_chars)))
    for line in range(len(class_cols[0])):
        print(" " + "|".join(col[line].ljust(max_col_width) for col in class_cols))
