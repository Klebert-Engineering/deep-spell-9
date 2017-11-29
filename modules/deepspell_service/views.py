
import os
import sys

import flask as fl

from . import app

# ====================[ Local Imports ]===================

from deepspell.models.extrapolator import DSLstmExtrapolator
from deepspell.models.discriminator import DSLstmDiscriminator, tokenize_class_annotated_characters
from deepspell.models.encoder import DSVariationalLstmAutoEncoder
from deepspell.token_lookup_space import DSTokenLookupSpace
from deepspell.baseline.symspell import DSSymSpellBaseline

# ====================[ Initialization ]==================

discriminator_model = None
extrapolator_model = None
corrector_model = None
featureset = None
hostname = ""
lowercase = False


def init(args):
    """
    Must be called from the importing file before app.run()!
    """
    global discriminator_model, extrapolator_model, corrector_model, featureset, hostname, lowercase
    app.config.update(args)
    discriminator_model = DSLstmDiscriminator(args["discriminator"])
    extrapolator_model = DSLstmExtrapolator(args["extrapolator"], extrapolation_beam_count=6)
    if args["corrector"].lower().strip() == "symspell":
        corrector_model = DSSymSpellBaseline(args["corrector_files"])
    elif args["corrector"]:
        corrector_model = DSVariationalLstmAutoEncoder(args["corrector"])
        corrector_model = DSTokenLookupSpace(corrector_model, args["corrector_files"])
    assert extrapolator_model.featureset.is_compatible(discriminator_model.featureset)
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
        with_correction=corrector_model is not None
    )


@app.route("/extrapolate")
def extrapolate():
    s = fl.request.args.get("s").lower().lstrip()
    if s:
        classes = discriminator_model.discriminate(featureset, s)
        best_classes = [col[0][0] for col in classes][:-1]
        completion = extrapolator_model.extrapolate(featureset, s, best_classes, 16)
        tokenization = tokenize_class_annotated_characters(s, classes)
        if corrector_model:
            for classname, token in tokenization.items():
                tokenization[classname] = [token]+corrector_model.match(token, k=3)
        return fl.jsonify({"discriminator": classes, "extrapolator": completion, "corrector": tokenization})
    return fl.jsonify("{}")
