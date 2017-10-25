
import json
import codecs
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/modules")

import deepspell_service

with codecs.open("service.json") as config_file:
    params = json.load(config_file)

deepspell_service.views.init(params)
deepspell_service.app.run(host=params["host"], port=params["port"])
