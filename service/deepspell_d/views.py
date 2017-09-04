
import flask as fl
import sys
import os
from . import app
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../modules"))

# ====================[ Local Imports ]===================

from deepspell.corpus import DSCorpus
from deepspell.extrapolator import DSLstmExtrapolator
from deepspell.discriminator import DSLstmDiscriminator

# ====================[ Initialization ]==================

discriminator_model = None
extrapolator_model = None
prediction_corpus = None


def init(args):
    """
    Must be called from the importing file before app.run()!
    """
    global discriminator_model, extrapolator_model, prediction_corpus
    app.config.update(args)
    prediction_corpus = DSCorpus(args["corpus"], "na")
    discriminator_model = DSLstmDiscriminator(args["discriminator"])
    extrapolator_model = DSLstmExtrapolator(args["extrapolator"])


# ========================[ Routes ]======================

@app.route("/")
def hello():
    return fl.render_template(
        "index.html",
        encoder_model_name=discriminator_model.name()+" / "+extrapolator_model.name())

# @app.route("/extrapolate")
# def most_similar_2_query():
#     s = fl.request.args.get("s")
#     print("Answering query '{}' ...".format(q))
#     qv = model.encode(nltk.word_tokenize(q))
#     n = int(fl.request.args.get("n"))
#     return fl.jsonify(dict({"ranking": embedded_corpus.most_similar_by_cosine(qv, n)}))


@app.route("/discriminate")
def discriminate():
    s = fl.request.args.get("s")
    result = discriminator_model.discriminate(prediction_corpus, s)
    return fl.jsonify({"sequence": result})
