"""Training loops for the three model kinds.

Deliberately framework-free: Adam, cross-entropy, periodic validation on a
held-out sampler seed, model-card persistence. Device selection prefers
CUDA, then Apple-Silicon MPS, then CPU — the same code runs on a GPU box.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
from torch import nn

from ..charset import DEFAULT_CHARSET, CharVocab, ClassVocab
from ..gazetteer import Gazetteer
from ..models import Completer, ModelCard, TokenEncoder, save_model
from ..models.tagger import Tagger
from ..sampling.grammar import PhraseGrammar
from .dataset import (
    IGNORE_INDEX,
    PhraseSampler,
    TokenPairSampler,
    completer_batch,
    encoder_batch,
    tagger_batch,
)

logger = logging.getLogger(__name__)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainSettings:
    steps: int = 2000
    batch_size: int = 64
    lr: float = 1e-3
    seed: int = 0
    val_every: int = 250
    val_batches: int = 8
    log_every: int = 50
    device: str = "auto"
    charset: str = DEFAULT_CHARSET
    lowercase: bool = True
    truncate_min: int | None = None  # tagger prefix truncation
    hparams: dict = field(default_factory=dict)


def _vocabs(gaz: Gazetteer, settings: TrainSettings) -> tuple[CharVocab, ClassVocab]:
    return (
        CharVocab(charset=settings.charset, lowercase=settings.lowercase),
        ClassVocab(names=tuple(gaz.classes())),
    )


def _loop(settings: TrainSettings, model: nn.Module, step_fn, val_fn) -> dict:
    device = pick_device(settings.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    torch.manual_seed(settings.seed)
    started = time.time()
    last_val: dict = {}
    for step in range(1, settings.steps + 1):
        model.train()
        loss = step_fn(device)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % settings.log_every == 0 or step == settings.steps:
            logger.info("step %d/%d loss %.4f", step, settings.steps, float(loss.detach()))
        if step % settings.val_every == 0 or step == settings.steps:
            model.eval()
            last_val = val_fn(device)
            logger.info("step %d validation: %s", step, last_val)
    last_val["train_seconds"] = round(time.time() - started, 1)
    last_val["device"] = str(device)
    return last_val


def train_tagger(
    gaz: Gazetteer, grammar: PhraseGrammar, out_base: str, settings: TrainSettings
) -> str:
    char_vocab, class_vocab = _vocabs(gaz, settings)
    hparams = {"emb_dim": 64, "hidden": 128, "layers": 2, "dropout": 0.1, **settings.hparams}
    model = Tagger(len(char_vocab), len(class_vocab), **hparams)
    truncate = settings.truncate_min if settings.truncate_min is not None else 3
    train_sampler = PhraseSampler(
        gaz, grammar, char_vocab, class_vocab, seed=settings.seed, truncate_min=truncate
    )
    val_sampler = PhraseSampler(
        gaz, grammar, char_vocab, class_vocab, seed=settings.seed + 10_000, truncate_min=truncate
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def step_fn(device: torch.device) -> torch.Tensor:
        ids, targets, lengths = tagger_batch(train_sampler, settings.batch_size, device)
        logits = model(ids, lengths)
        return criterion(logits.flatten(0, 1), targets.flatten())

    @torch.inference_mode()
    def val_fn(device: torch.device) -> dict:
        correct = total = 0
        for _ in range(settings.val_batches):
            ids, targets, lengths = tagger_batch(val_sampler, settings.batch_size, device)
            predictions = model(ids, lengths).argmax(dim=-1)
            valid = targets != IGNORE_INDEX
            correct += int((predictions[valid] == targets[valid]).sum())
            total += int(valid.sum())
        return {"char_accuracy": round(correct / max(total, 1), 4)}

    metrics = _loop(settings, model, step_fn, val_fn)
    card = ModelCard(
        kind="tagger",
        charset=settings.charset,
        lowercase=settings.lowercase,
        classes=list(class_vocab.names),
        hparams=hparams,
        training={"corpus": gaz.path, "steps": settings.steps, "batch_size": settings.batch_size, **metrics},
    )
    return str(save_model(model, card, out_base))


def train_completer(
    gaz: Gazetteer, grammar: PhraseGrammar, out_base: str, settings: TrainSettings
) -> str:
    char_vocab, class_vocab = _vocabs(gaz, settings)
    hparams = {"emb_dim": 64, "hidden": 256, "layers": 2, "dropout": 0.1, **settings.hparams}
    model = Completer(len(char_vocab), len(class_vocab), **hparams)
    train_sampler = PhraseSampler(gaz, grammar, char_vocab, class_vocab, seed=settings.seed)
    val_sampler = PhraseSampler(gaz, grammar, char_vocab, class_vocab, seed=settings.seed + 10_000)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def losses(sampler: PhraseSampler, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        in_chars, in_classes, tgt_chars, tgt_classes, lengths = completer_batch(
            sampler, settings.batch_size, device
        )
        char_logits, class_logits, _ = model(in_chars, in_classes, lengths)
        char_loss = criterion(char_logits.flatten(0, 1), tgt_chars.flatten())
        class_loss = criterion(class_logits.flatten(0, 1), tgt_classes.flatten())
        return char_loss, class_loss

    def step_fn(device: torch.device) -> torch.Tensor:
        char_loss, class_loss = losses(train_sampler, device)
        return char_loss + class_loss

    @torch.inference_mode()
    def val_fn(device: torch.device) -> dict:
        char_total = class_total = 0.0
        for _ in range(settings.val_batches):
            char_loss, class_loss = losses(val_sampler, device)
            char_total += float(char_loss)
            class_total += float(class_loss)
        return {
            "val_char_loss": round(char_total / settings.val_batches, 4),
            "val_class_loss": round(class_total / settings.val_batches, 4),
        }

    metrics = _loop(settings, model, step_fn, val_fn)
    card = ModelCard(
        kind="completer",
        charset=settings.charset,
        lowercase=settings.lowercase,
        classes=list(class_vocab.names),
        hparams=hparams,
        training={"corpus": gaz.path, "steps": settings.steps, "batch_size": settings.batch_size, **metrics},
    )
    return str(save_model(model, card, out_base))


def train_encoder(
    gaz: Gazetteer,
    out_base: str,
    settings: TrainSettings,
    corruption_mean: float = 1.0,
    corruption_stddev: float = 0.5,
    temperature: float = 0.1,
) -> str:
    char_vocab = CharVocab(charset=settings.charset, lowercase=settings.lowercase)
    hparams = {"emb_dim": 48, "hidden": 128, "layers": 1, "out_dim": 32, **settings.hparams}
    model = TokenEncoder(len(char_vocab), **hparams)
    train_sampler = TokenPairSampler(
        gaz, char_vocab, seed=settings.seed, corruption_mean=corruption_mean,
        corruption_stddev=corruption_stddev,
    )
    val_sampler = TokenPairSampler(
        gaz, char_vocab, seed=settings.seed + 10_000, corruption_mean=corruption_mean,
        corruption_stddev=corruption_stddev,
    )
    criterion = nn.CrossEntropyLoss()

    def info_nce(sampler: TokenPairSampler, device: torch.device) -> torch.Tensor:
        corrupt_ids, corrupt_lengths, clean_ids, clean_lengths = encoder_batch(
            sampler, settings.batch_size, device
        )
        anchors = model(corrupt_ids, corrupt_lengths)
        positives = model(clean_ids, clean_lengths)
        logits = anchors @ positives.T / temperature
        labels = torch.arange(logits.shape[0], device=device)
        return criterion(logits, labels)

    def step_fn(device: torch.device) -> torch.Tensor:
        return info_nce(train_sampler, device)

    @torch.inference_mode()
    def val_fn(device: torch.device) -> dict:
        hits = total = 0
        for _ in range(settings.val_batches):
            corrupt_ids, corrupt_lengths, clean_ids, clean_lengths = encoder_batch(
                val_sampler, settings.batch_size, device
            )
            similarity = model(corrupt_ids, corrupt_lengths) @ model(clean_ids, clean_lengths).T
            hits += int((similarity.argmax(dim=1) == torch.arange(similarity.shape[0], device=device)).sum())
            total += similarity.shape[0]
        return {"val_retrieval_top1": round(hits / max(total, 1), 4)}

    metrics = _loop(settings, model, step_fn, val_fn)
    card = ModelCard(
        kind="encoder",
        charset=settings.charset,
        lowercase=settings.lowercase,
        classes=[],
        hparams=hparams,
        training={
            "corpus": gaz.path,
            "steps": settings.steps,
            "batch_size": settings.batch_size,
            "corruption_mean": corruption_mean,
            "corruption_stddev": corruption_stddev,
            **metrics,
        },
    )
    return str(save_model(model, card, out_base))
