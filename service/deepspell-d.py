
import deepspell_d
import json
import codecs

with codecs.open("deepspell-d.json") as config_file:
    params = json.load(config_file)

deepspell_d.views.init(params)
deepspell_d.app.run(host=params["host"], port=params["port"])
