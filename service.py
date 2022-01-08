import json
import codecs
import deepspell_service

with codecs.open("service.json") as config_file:
    params = json.load(config_file)

deepspell_service.views.init(params)
app = deepspell_service.app

if __name__ == "__main__":
    deepspell_service.app.run()
