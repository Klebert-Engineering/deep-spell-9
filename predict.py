# (C) 2017 Klebert Engineering GmbH

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/modules")

from deepspell.corpus import DSCorpus
from deepspell.extrapolator import DSLstmExtrapolator
from deepspell.discriminator import DSLstmDiscriminator

extrapolator_model = DSLstmExtrapolator("models/deepsp_extra-v1_na_lr003_dec50_bat4096_128-128.json", "logs")
# extrapolator_model = DSLstmExtrapolator("models/deepsp_extra-v2_na_lr003_dec50_bat4096_128-128.json", "logs")
# extrapolator_model = DSLstmExtrapolator("models/deepsp_extra-v2_na_lr003_dec50_bat3072_128-128-128.json", "logs")
discriminator_model = DSLstmDiscriminator("models/deepsp_discr-v1_na_lr003_dec50_bat3072_fw128-128_bw128.json", "logs")
assert extrapolator_model.featureset.is_compatible(discriminator_model.featureset)
featureset = extrapolator_model.featureset

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
           
 -> To obtain a class discrimination, enter any string that is
    not 'q' and does not contain '~'.
      
Have fun!

""".format(
    "\n".join("    {}: {}".format(class_name, class_id) for class_name, class_id in featureset.class_ids.items())
))

while True:
    user_command = input("> ")
    if user_command == "q":
        exit(0)

    completion_chars = None
    completion_classes = None

    if "~" in user_command:
        parts = user_command.split("~")

        prefix_chars, prefix_classes = parts
        prefix_classes = prefix_classes.strip()
        if len(prefix_chars) == len(prefix_classes):
            prefix_class_names = []
            for cl in prefix_classes:
                cl_name = featureset.class_name_for_id(int(cl))
                if cl_name:
                    prefix_class_names.append(cl_name)
                else:
                    print("{} is not a valid class id!".format(cl))
                    prefix_class_names = []
                    break
        elif not prefix_classes:
            prefix_class_names = [col[0][0] for col in discriminator_model.discriminate(featureset, prefix_chars)][:-1]
            print("Prefix classes:", prefix_class_names)
        else:
            print("The command part after '~' must be either of zero or of equal length!")

        if len(prefix_class_names) != len(prefix_chars):
            print("Invalid prefix classes.")
            continue

        completion_chars, completion_classes = extrapolator_model.extrapolate(
            featureset,
            prefix_chars,
            prefix_class_names, 8)

    else:
        completion_classes = discriminator_model.discriminate(featureset, user_command)
        completion_chars = []

    def pct_(f):
        return str(int(f*100.0))

    char_cols = [[] for _ in range(len(completion_classes))]
    class_cols = [[] for _ in range(len(completion_classes))]
    for t in range(len(completion_classes)):
        if completion_chars:
            for i in range(3):
                char_cols[t].append(" {} {}% ".format(completion_chars[t][i][0], pct_(completion_chars[t][i][1])))
        for i in range(3):
            class_cols[t].append(" {} {}% ".format(completion_classes[t][i][0][:2], pct_(completion_classes[t][i][1])))
    max_col_width = max(len(s) for col in class_cols + char_cols for s in col)
    if completion_chars:
        for line in range(len(char_cols[0])):
            print(" " + "|".join(col[line].ljust(max_col_width) for col in char_cols))
        print(" " + "|".join(["-"*max_col_width] * len(completion_chars)))
    for line in range(len(class_cols[0])):
        print(" " + "|".join(col[line].ljust(max_col_width) for col in class_cols))
