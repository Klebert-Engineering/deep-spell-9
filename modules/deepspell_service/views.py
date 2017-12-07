
import os
import sys

import flask as fl
import time

from collections import defaultdict
from . import app

# ====================[ Local Imports ]===================

from deepspell.models.extrapolator import DSLstmExtrapolator
from deepspell.models.discriminator import DSLstmDiscriminator, tokenize_class_annotated_characters, extract_best_class_sequence
from deepspell.models.encoder import DSVariationalLstmAutoEncoder
from deepspell.token_lookup_space import DSTokenLookupSpace
from deepspell.baseline.symspell import DSSymSpellBaseline
from deepspell.ftsdb import DSFtsDatabaseConnection

# ====================[ Initialization ]==================

discriminator_model = None
extrapolator_model = None
corrector_model = None
featureset = None
hostname = ""
lowercase = False
fts_lookup_db = None


def init(args):
    """
    Must be called from the importing file before app.run()!
    """
    global discriminator_model, extrapolator_model, corrector_model, featureset, hostname, lowercase, fts_lookup_db
    app.config.update(args)
    discriminator_model = DSLstmDiscriminator(args["discriminator"])
    extrapolator_model = DSLstmExtrapolator(args["extrapolator"], extrapolation_beam_count=6)
    if args["corrector"].lower().strip() == "symspell":
        corrector_model = DSSymSpellBaseline(args["corrector_files"])
    elif args["corrector"]:
        corrector_model = DSVariationalLstmAutoEncoder(args["corrector"])
        corrector_model = DSTokenLookupSpace(corrector_model, args["corrector_files"])
    assert extrapolator_model.featureset.is_compatible(discriminator_model.featureset)
    if args["fts_db"]:
        fts_lookup_db = DSFtsDatabaseConnection(**args["fts_db"])
    featureset = extrapolator_model.featureset
    hostname = args["hostname"]+":"+str(args["port"])
    lowercase = args["lowercase"]


# ========================[ Routes ]======================


@app.route("/")
def hello():
    return fl.render_template(
        "index.html",
        encoder_model_name=discriminator_model.name()+" / "+extrapolator_model.name(),
        hostname=hostname,
        with_correction=corrector_model is not None,
        with_ftslookup=fts_lookup_db is not None
    )


@app.route("/extrapolate")
def extrapolate():
    s = fl.request.args.get("s").lower().lstrip()
    timings = {}

    def stoptime(key):
        if key in timings:
            timings[key] = time.time()-timings[key]
        else:
            timings[key] = time.time()

    if s:
        # -- 1.) Discriminate token classes, strip final EOL class
        stoptime("classification")
        classes = discriminator_model.discriminate(featureset, s)[:-1]
        best_classes = extract_best_class_sequence(s, classes)
        stoptime("classification")

        # -- 2.) Get completion alternatives
        stoptime("completion")
        completion = extrapolator_model.extrapolate(featureset, s, best_classes, 16)
        stoptime("completion")

        # -- 3.) Tokenize (with best completion appended if it completes the last token's class)
        tokenization_classes = classes[:]
        tokenization_string = s[:]
        if completion[0][1][0] == best_classes[-1]:
            extrapolation_class_distribution = [
                [cl, 1. if cl == best_classes[-1] else 0.]
                for cl in featureset.class_ids]
            tokenization_classes += [extrapolation_class_distribution] * len(completion[0][0])
            tokenization_string += completion[0][0]
        tokenization = tokenize_class_annotated_characters(tokenization_string, tokenization_classes)

        # -- 4.) Correct the tokens
        stoptime("correction")
        if corrector_model:
            for classname, token in tokenization.items():
                tokenization[classname] = [token]+corrector_model.match(token, k=3)
        stoptime("correction")

        return fl.jsonify({
            "discriminator": classes,
            "extrapolator": completion,
            "corrector": tokenization,
            "timings": timings})
    return fl.jsonify("{}")


@app.route("/lookup")
def lookup():
    if not fts_lookup_db:
        return fl.jsonify([])
    criteria = {key: value for key, value in fl.request.args.items()}
    print(criteria)
    n = criteria.pop("n", 10)
    return fl.jsonify(fts_lookup_db.lookup_fts_entries(limit=n, **criteria))
