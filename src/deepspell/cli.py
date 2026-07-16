"""``ds9`` command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from . import __version__
from .charset import CharVocab, ClassVocab
from .gazetteer import Gazetteer

app = typer.Typer(help="Deep-Spell-9 v2: neural geographic query processing.", no_args_is_help=True)
data_app = typer.Typer(help="Gazetteer construction and derived artifacts.", no_args_is_help=True)
train_app = typer.Typer(help="Model training.", no_args_is_help=True)
eval_app = typer.Typer(help="Model evaluation.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


@app.callback()
def main() -> None:
    """Deep-Spell-9 v2."""


@app.command()
def version() -> None:
    typer.echo(__version__)


# --------------------------------------------------------------------- data


@data_app.command("import-legacy-tsv")
def data_import_legacy_tsv(tsv: Path, gazetteer: Path) -> None:
    """Import a v1 training TSV (class/id/name/parent rows) into a gazetteer."""
    from .gazetteer.legacy_tsv import import_legacy_tsv

    with Gazetteer(gazetteer) as gaz:
        count = import_legacy_tsv(gaz, tsv)
    typer.echo(f"imported {count} tokens into {gazetteer}")


@data_app.command("fetch-geonames")
def data_fetch_geonames(directory: Path, cities: str = "cities500") -> None:
    """Download GeoNames dumps (countryInfo, admin1 codes, cities)."""
    from .gazetteer.geonames import fetch_geonames

    fetch_geonames(directory, cities_file=cities)
    typer.echo(f"geonames files ready under {directory}")


@data_app.command("import-geonames")
def data_import_geonames(directory: Path, gazetteer: Path, cities: str = "cities500") -> None:
    """Import GeoNames dumps (COUNTRY/STATE/CITY) into a gazetteer."""
    from .gazetteer.geonames import import_geonames

    with Gazetteer(gazetteer) as gaz:
        count = import_geonames(gaz, directory, cities_file=cities)
    typer.echo(f"imported {count} tokens into {gazetteer}")


@data_app.command("info")
def data_info(gazetteer: Path) -> None:
    """Token counts per class."""
    with Gazetteer(gazetteer, readonly=True) as gaz:
        for class_name in gaz.classes():
            typer.echo(f"{class_name:10s} {gaz.count(class_name)}")
        typer.echo(f"{'TOTAL':10s} {gaz.count()}")


@data_app.command("build-lookup")
def data_build_lookup(gazetteer: Path, out: Path) -> None:
    """Build the FTS5 lookup index served by /api/lookup."""
    from .lookup import build_lookup_index

    with Gazetteer(gazetteer, readonly=True) as gaz:
        build_lookup_index(gaz, out)
    typer.echo(f"lookup index written to {out}")


@data_app.command("build-symspell")
def data_build_symspell(gazetteer: Path, out: Path) -> None:
    """Build the symspell frequency dictionary for the default corrector."""
    from .correct.symspell import build_dictionary

    with Gazetteer(gazetteer, readonly=True) as gaz:
        build_dictionary(gaz, out)
    typer.echo(f"symspell dictionary written to {out}")


@data_app.command("build-embedding-space")
def data_build_embedding_space(
    encoder: Path,
    gazetteer: Path,
    out: Path,
    class_name: str | None = typer.Option(None, "--class", help="restrict to one token class"),
) -> None:
    """Encode gazetteer names into an embedding-corrector space (.npz + .tokens)."""
    from .correct.embedding import EmbeddingCorrector
    from .models import load_model

    encoder_model, card = load_model(encoder)
    with Gazetteer(gazetteer, readonly=True) as gaz:
        names = [name for name, _ in gaz.iter_names(class_name)]
    space = EmbeddingCorrector.build(encoder_model, card.char_vocab(), names)
    space.save(out)
    typer.echo(f"embedded {len(space.tokens)} tokens to {out}.npz/.tokens")


# -------------------------------------------------------------------- train


def _settings(steps, batch_size, lr, seed, device, hidden, layers, emb_dim):
    from .train import TrainSettings

    hparams = {}
    if hidden is not None:
        hparams["hidden"] = hidden
    if layers is not None:
        hparams["layers"] = layers
    if emb_dim is not None:
        hparams["emb_dim"] = emb_dim
    return TrainSettings(
        steps=steps, batch_size=batch_size, lr=lr, seed=seed, device=device, hparams=hparams
    )


_STEPS = typer.Option(2000, help="training steps")
_BATCH = typer.Option(64, help="batch size")
_LR = typer.Option(1e-3, help="learning rate")
_SEED = typer.Option(0)
_DEVICE = typer.Option("auto", help="auto | cpu | cuda | mps")
_HIDDEN = typer.Option(None, help="LSTM hidden size")
_LAYERS = typer.Option(None, help="LSTM layers")
_EMB = typer.Option(None, help="embedding dim")


@train_app.command("tagger")
def train_tagger_cmd(
    gazetteer: Path,
    grammar: Path,
    out: Path,
    steps: int = _STEPS,
    batch_size: int = _BATCH,
    lr: float = _LR,
    seed: int = _SEED,
    device: str = _DEVICE,
    hidden: int | None = _HIDDEN,
    layers: int | None = _LAYERS,
    emb_dim: int | None = _EMB,
) -> None:
    from .sampling.grammar import PhraseGrammar
    from .train import train_tagger

    with Gazetteer(gazetteer, readonly=True) as gaz:
        path = train_tagger(
            gaz, PhraseGrammar.from_file(grammar), str(out),
            _settings(steps, batch_size, lr, seed, device, hidden, layers, emb_dim),
        )
    typer.echo(f"saved {path}")


@train_app.command("completer")
def train_completer_cmd(
    gazetteer: Path,
    grammar: Path,
    out: Path,
    steps: int = _STEPS,
    batch_size: int = _BATCH,
    lr: float = _LR,
    seed: int = _SEED,
    device: str = _DEVICE,
    hidden: int | None = _HIDDEN,
    layers: int | None = _LAYERS,
    emb_dim: int | None = _EMB,
) -> None:
    from .sampling.grammar import PhraseGrammar
    from .train import train_completer

    with Gazetteer(gazetteer, readonly=True) as gaz:
        path = train_completer(
            gaz, PhraseGrammar.from_file(grammar), str(out),
            _settings(steps, batch_size, lr, seed, device, hidden, layers, emb_dim),
        )
    typer.echo(f"saved {path}")


@train_app.command("encoder")
def train_encoder_cmd(
    gazetteer: Path,
    out: Path,
    steps: int = _STEPS,
    batch_size: int = _BATCH,
    lr: float = _LR,
    seed: int = _SEED,
    device: str = _DEVICE,
    hidden: int | None = _HIDDEN,
    layers: int | None = _LAYERS,
    emb_dim: int | None = _EMB,
    corruption_mean: float = typer.Option(1.0),
    corruption_stddev: float = typer.Option(0.5),
) -> None:
    from .train import train_encoder

    with Gazetteer(gazetteer, readonly=True) as gaz:
        path = train_encoder(
            gaz, str(out),
            _settings(steps, batch_size, lr, seed, device, hidden, layers, emb_dim),
            corruption_mean=corruption_mean, corruption_stddev=corruption_stddev,
        )
    typer.echo(f"saved {path}")


# --------------------------------------------------------------------- eval


@eval_app.command("tagger")
def eval_tagger_cmd(
    model: Path, gazetteer: Path, grammar: Path, samples: int = 200, seed: int = 99
) -> None:
    from .models import load_model
    from .sampling.grammar import PhraseGrammar
    from .train.dataset import PhraseSampler
    from .train.evaluate import evaluate_tagger

    tagger, card = load_model(model)
    with Gazetteer(gazetteer, readonly=True) as gaz:
        sampler = PhraseSampler(
            gaz, PhraseGrammar.from_file(grammar), card.char_vocab(), card.class_vocab(),
            seed=seed, truncate_min=3,
        )
        metrics = evaluate_tagger(tagger, sampler, card.char_vocab(), card.class_vocab(), samples)
        typer.echo(json.dumps(metrics))


@eval_app.command("completer")
def eval_completer_cmd(
    model: Path, gazetteer: Path, grammar: Path, samples: int = 100, seed: int = 99, beam: int = 6
) -> None:
    from .models import load_model
    from .sampling.grammar import PhraseGrammar
    from .train.dataset import PhraseSampler
    from .train.evaluate import evaluate_completer

    completer, card = load_model(model)
    with Gazetteer(gazetteer, readonly=True) as gaz:
        sampler = PhraseSampler(
            gaz, PhraseGrammar.from_file(grammar), card.char_vocab(), card.class_vocab(), seed=seed
        )
        typer.echo(
            json.dumps(
                evaluate_completer(
                    completer, sampler, card.char_vocab(), card.class_vocab(), samples, beam_width=beam
                )
            )
        )


@eval_app.command("corrector")
def eval_corrector_cmd(
    gazetteer: Path,
    backend: str = typer.Option("symspell", help="symspell | embedding"),
    dictionary: Path | None = typer.Option(None, help="symspell dictionary path"),
    encoder: Path | None = typer.Option(None, help="encoder model card (embedding backend)"),
    space: Path | None = typer.Option(None, help="embedding space base path"),
    samples: int = 200,
    k: int = 3,
    seed: int = 99,
) -> None:
    from .train.evaluate import evaluate_corrector

    vocab = CharVocab()
    if backend == "symspell":
        from .correct import SymSpellCorrector

        corrector = SymSpellCorrector(dictionary)
    else:
        from .correct import EmbeddingCorrector
        from .models import load_model

        encoder_model, card = load_model(encoder)
        corrector = EmbeddingCorrector.load(encoder_model, card.char_vocab(), space)
        vocab = card.char_vocab()
    with Gazetteer(gazetteer, readonly=True) as gaz:
        names = [vocab.normalize(name) for name, _ in gaz.iter_names() if vocab.normalize(name)]
    typer.echo(json.dumps(evaluate_corrector(corrector, names, seed=seed, samples=samples, k=k)))


# -------------------------------------------------------------------- serve


@app.command()
def serve(
    config: Path,
    host: str = "0.0.0.0",
    port: int = 8091,
) -> None:
    """Run the web service from a JSON config file."""
    import uvicorn

    from .service import ServiceConfig, create_app

    uvicorn.run(create_app(config=ServiceConfig.from_json(config)), host=host, port=port)


@app.command()
def demo(tagger: Path, completer: Path) -> None:
    """Interactive REPL: classify and complete queries (successor of predict.py)."""
    from .decode import beam_search, best_class_sequence
    from .models import load_model

    tagger_model, tagger_card = load_model(tagger)
    completer_model, _ = load_model(completer)
    char_vocab: CharVocab = tagger_card.char_vocab()
    class_vocab: ClassVocab = tagger_card.class_vocab()
    typer.echo("Enter a query prefix ('q' to quit):")
    while True:
        try:
            query = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip() == "q":
            break
        text = char_vocab.normalize(query)
        if not text:
            continue
        probs = tagger_model.predict(text, char_vocab, class_vocab)
        best = best_class_sequence(text, probs)
        typer.echo("  classes: " + " ".join(f"{ch}:{cls[:2]}" for ch, cls in zip(text, best)))
        completions = beam_search(
            completer_model, char_vocab, class_vocab, text,
            [class_vocab.encode(name) for name in best],
        )
        for completion in completions:
            typer.echo(f"  {text}[{completion.text}]  ({completion.class_name}, {completion.logprob:.2f})")


if __name__ == "__main__":
    app()
