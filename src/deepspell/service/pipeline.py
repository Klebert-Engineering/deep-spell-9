"""Query-processing pipeline: tag -> complete -> tokenize -> correct (-> lookup)."""

from __future__ import annotations

import time

from ..correct import Corrector, EmbeddingCorrector, SymSpellCorrector
from ..decode import beam_search, best_class_sequence, split_by_class
from ..lookup import FtsLookup
from ..models import load_model
from .config import ServiceConfig
from .schemas import (
    ClassProb,
    CompleteResponse,
    CompletionOut,
    CorrectionOut,
    SuggestionOut,
)


class Pipeline:
    def __init__(
        self,
        tagger,
        tagger_card,
        completer,
        completer_card,
        corrector: Corrector | None = None,
        lookup: FtsLookup | None = None,
        beam_width: int = 6,
        max_completion_length: int = 16,
        corrections: int = 3,
    ):
        if tagger_card.charset != completer_card.charset or tagger_card.classes != completer_card.classes:
            raise ValueError("tagger and completer model cards are incompatible")
        self.tagger = tagger
        self.completer = completer
        self.char_vocab = tagger_card.char_vocab()
        self.class_vocab = tagger_card.class_vocab()
        self.corrector = corrector
        self.lookup_db = lookup
        self.beam_width = beam_width
        self.max_completion_length = max_completion_length
        self.corrections = corrections

    @classmethod
    def from_config(cls, config: ServiceConfig) -> Pipeline:
        tagger, tagger_card = load_model(config.tagger, device=config.device)
        completer, completer_card = load_model(config.completer, device=config.device)

        corrector: Corrector | None = None
        if config.corrector == "symspell":
            corrector = SymSpellCorrector(config.symspell_dictionary)
        elif config.corrector == "embedding":
            encoder, encoder_card = load_model(config.encoder, device=config.device)
            corrector = EmbeddingCorrector.load(
                encoder, encoder_card.char_vocab(), config.embedding_space
            )

        lookup = None
        if config.lookup:
            if config.lookup_kind == "nds":
                lookup = FtsLookup.nds({"path": str(config.lookup), **(config.nds or {})})
            else:
                lookup = FtsLookup(config.lookup)

        return cls(
            tagger, tagger_card, completer, completer_card,
            corrector=corrector, lookup=lookup,
            beam_width=config.beam_width,
            max_completion_length=config.max_completion_length,
            corrections=config.corrections,
        )

    def complete(self, query: str) -> CompleteResponse:
        timings: dict[str, float] = {}
        text = self.char_vocab.normalize(query.lstrip())

        started = time.perf_counter()
        probs = self.tagger.predict(text, self.char_vocab, self.class_vocab)
        best = best_class_sequence(text, probs)
        timings["classification"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        completions = []
        if text:
            completions = beam_search(
                self.completer,
                self.char_vocab,
                self.class_vocab,
                text,
                [self.class_vocab.encode(name) for name in best],
                beam_width=self.beam_width,
                max_len=self.max_completion_length,
            )
        timings["completion"] = (time.perf_counter() - started) * 1000

        # -- tokenize, with the top completion appended if it continues the last token
        split_text, split_classes = text, list(best)
        if completions and completions[0].text and best and completions[0].class_name == best[-1]:
            split_text += completions[0].text
            split_classes += [completions[0].class_name] * len(completions[0].text)
        tokens = split_by_class(split_text, split_classes)

        started = time.perf_counter()
        corrections: dict[str, CorrectionOut] = {}
        if self.corrector is not None:
            for class_name, token in tokens.items():
                matches = self.corrector.match(token, k=self.corrections)
                corrections[class_name] = CorrectionOut(
                    input=token,
                    suggestions=[SuggestionOut(text=t, distance=d) for t, d in matches],
                )
        timings["correction"] = (time.perf_counter() - started) * 1000

        return CompleteResponse(
            query=text,
            classes=[[ClassProb(name=n, p=p) for n, p in char_probs] for char_probs in probs],
            completions=[
                CompletionOut(text=c.text, class_name=c.class_name, logprob=c.logprob)
                for c in completions
            ],
            tokens=tokens,
            corrections=corrections,
            timings_ms={key: round(value, 2) for key, value in timings.items()},
        )

    def lookup(self, criteria: dict[str, str], limit: int = 10) -> list[dict]:
        if self.lookup_db is None:
            return []
        return self.lookup_db.query(criteria, limit=limit)
