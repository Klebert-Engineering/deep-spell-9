import json
import codecs
import deepspell_service
import os

config_file_path = "service.json"
if "SERVICE_CONFIG" in os.environ:
    config_file_path = os.environ["SERVICE_CONFIG"]

with codecs.open(config_file_path) as config_file:
    params = json.load(config_file)

deepspell_service.views.init(params)
app = deepspell_service.app

if __name__ == "__main__":
    deepspell_service.app.run()
