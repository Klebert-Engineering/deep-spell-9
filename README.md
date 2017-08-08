deep-spell-v1.0rc
=================

The DeepSpell project contains implementations of various neural algorithms,
which enable natural auto-suggestion via deep neural networks.

## Dependencies

Mandatory:
* python 3.x
* tensorflow >= 1.3
* unidecode
* flask
* scipy

Optional:
* pygtrie (for FTS5 baseline)
* hat_trie (for generating Symspell DAWG)
* DAWG (for generating/loading Symspell DAWG)

## Scripts

### `python3 service.py`

Launches the Deep Spell demonstrator web interface from the
`modules/deepspell_service` flask module. Use the adjacent
`service.json` config file to set the served port and models,
among other options.
