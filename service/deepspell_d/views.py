
import flask as fl
import sys
import os
from . import app
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../modules"))

# ====================[ Local Imports ]===================

from deepspell.extrapolator import DSLstmExtrapolator
from deepspell.discriminator import DSLstmDiscriminator

# ====================[ Initialization ]==================

discriminator_model = None
extrapolator_model = None
featureset = None


def init(args):
    """
    Must be called from the importing file before app.run()!
    """
    global discriminator_model, extrapolator_model, featureset
    app.config.update(args)
    discriminator_model = DSLstmDiscriminator(args["discriminator"])
    extrapolator_model = DSLstmExtrapolator(args["extrapolator"])
    assert extrapolator_model.featureset.is_compatible(discriminator_model.featureset)
    featureset = extrapolator_model.featureset


# ========================[ Routes ]======================

@app.route("/")
def hello():
    return fl.render_template(
        "index.html",
        encoder_model_name=discriminator_model.name()+" / "+extrapolator_model.name())


@app.route("/extrapolate")
def extrapolate():
    s = fl.request.args.get("s")
    if not s:
        return fl.jsonify({"sequence": []})
    classes = [col[0][0] for col in discriminator_model.discriminate(featureset, s)][:-1]
    result = extrapolator_model.extrapolate(featureset, s, classes, 16)
    return fl.jsonify({"sequence": result})


@app.route("/discriminate")
def discriminate():
    s = fl.request.args.get("s")
    s = fl.request.args.get("s")
    if not s:
        return fl.jsonify({"sequence": []})
    result = discriminator_model.discriminate(featureset, s)
    return fl.jsonify({"sequence": result})

