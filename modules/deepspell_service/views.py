
import os
import sys

import flask as fl

from . import app

# ====================[ Local Imports ]===================

from deepspell.models.extrapolator import DSLstmExtrapolator
from deepspell.models.discriminator import DSLstmDiscriminator

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
    global discriminator_model, extrapolator_model, featureset, hostname, lowercase
    app.config.update(args)
    discriminator_model = DSLstmDiscriminator(args["discriminator"])
    extrapolator_model = DSLstmExtrapolator(args["extrapolator"], extrapolation_beam_count=6)
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
        hostname=hostname
    )


@app.route("/extrapolate")
def extrapolate():
    s = fl.request.args.get("s").lower()
    if not s:
        return fl.jsonify({"sequence": []})
    classes = discriminator_model.discriminate(featureset, s)
    best_classes = [col[0][0] for col in classes][:-1]
    completion = extrapolator_model.extrapolate(featureset, s, best_classes, 16)
    return fl.jsonify({"discriminator": classes, "extrapolator": completion})


@app.route("/discriminate")
def discriminate():
    s = fl.request.args.get("s").lower()
    if not s:
        return fl.jsonify({"sequence": []})
    result = discriminator_model.discriminate(featureset, s)
    return fl.jsonify({"sequence": result})

