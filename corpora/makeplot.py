# (C) 2017 Klebert Engineering GmbH

import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import cKDTree
from collections import defaultdict
import plotly.offline as py
import plotly.graph_objs as go
import codecs
import random
import gensim
import json
import pickle

print("Rendering plot summaries.")

num_plots = 1
m = pickle.load(open("deepspell_data_north_america_cities.kdtree", "rb"))
tokens = [token.strip() for token in open("deepspell_data_north_america_cities.tokens")]
dims = 2
tsne = TSNE(perplexity=30, init='pca', n_components=dims, n_iter=20000, learning_rate=250, method='exact', early_exaggeration=10.0, metric='cosine')

print("Running t-SNE")
low_dim_embs = tsne.fit_transform(m.data)

# compile samples per token cluster
# dict of {name:[[text], [x],[y],[z]]}
print("Compiling token points per cluster ...")
coord_list = []
for i in range(len(low_dim_embs)):
    token = tokens[i]
    coords = [[], [], []]
    assert coords
    coords[0].append(token)
    coords[1].append(low_dim_embs[i][0])
    coords[2].append(low_dim_embs[i][1])

print("\nCreating scatter plots ...")
data = [go.Scatter(
    text=coords[0],
    x=coords[1],
    y=coords[2],
    name="NDS FTS Lexical Lookup Space",
    hoverinfo="text+name",
    mode='markers' # ,
    #marker=dict(
    #    color='rgb(%i, %i, %i)' % (colors[index][0], colors[index][1], colors[index][2]),
    #    size=5,
    #    symbol='circle',
    #    line=dict(width=0),
    #    opacity=1.0
    #)
)]

layout = go.Layout(
    hovermode="x+y",
    title="NDS FTS Lexical Lookup Space / 8 dims.",
    margin=dict(l=0, r=0, b=0, t=0 ))

py.plot({"data":data, "layout":layout})

