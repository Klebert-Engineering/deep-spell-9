"""Beam-search completion decoding.

Replaces the v1 in-graph ``tf.while_loop`` beam search with ~80 lines of
Python. Semantics follow v1: all beams adopt the argmax class of the first
predicted step, and a beam finishes when the predicted class switches (the
switching character is not emitted) or the end-of-sequence character is
predicted. A completion therefore never crosses a token-class boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..charset import BOS, EOS, PAD, UNK, CharVocab, ClassVocab
from ..models.completer import Completer


@dataclass(frozen=True)
class Completion:
    text: str
    class_name: str
    logprob: float


@torch.inference_mode()
def beam_search(
    completer: Completer,
    char_vocab: CharVocab,
    class_vocab: ClassVocab,
    prefix_text: str,
    prefix_class_ids: list[int],
    beam_width: int = 6,
    max_len: int = 16,
) -> list[Completion]:
    """Rank completions of *normalized* ``prefix_text``.

    ``prefix_class_ids`` must hold one class id per prefix character (from
    the tagger's best class sequence).
    """
    if not prefix_text or len(prefix_text) != len(prefix_class_ids):
        return []
    device = next(completer.parameters()).device

    char_ids = torch.tensor([char_vocab.encode(prefix_text, add_bos=True)], device=device)
    class_ids = torch.tensor([[prefix_class_ids[0]] + prefix_class_ids], device=device)
    char_logits, class_logits, state = completer(char_ids, class_ids)

    mask = torch.full((len(char_vocab),), 0.0, device=device)
    mask[[PAD, BOS, UNK]] = float("-inf")

    # -- first step: all beams adopt the argmax class (v1 behaviour)
    beam_class = int(class_logits[0, -1].argmax())
    char_logp = torch.log_softmax(char_logits[0, -1] + mask, dim=-1)
    k = min(beam_width, char_logp.shape[-1])
    top_logp, top_chars = torch.topk(char_logp, k)

    if beam_class == class_vocab.eol_id:
        return [Completion("", class_vocab.decode(beam_class), float(top_logp[0]))]

    hidden = (state[0].repeat_interleave(k, dim=1), state[1].repeat_interleave(k, dim=1))
    texts: list[list[int]] = [[int(c)] for c in top_chars]
    logprobs = top_logp.clone()
    finished: list[Completion] = []
    active = [i for i in range(k) if int(top_chars[i]) != EOS]
    for i in range(k):
        if int(top_chars[i]) == EOS:
            finished.append(Completion("", class_vocab.decode(beam_class), float(logprobs[i])))
    texts = [texts[i] for i in active]
    logprobs = logprobs[active]
    hidden = (hidden[0][:, active], hidden[1][:, active])

    for _ in range(max_len - 1):
        if not texts:
            break
        n = len(texts)
        last_chars = torch.tensor([[t[-1]] for t in texts], device=device)
        last_classes = torch.full((n, 1), beam_class, dtype=torch.long, device=device)
        char_logits, class_logits, state = completer(last_chars, last_classes, state=hidden)
        step_logp = torch.log_softmax(char_logits[:, 0] + mask, dim=-1)
        pred_classes = class_logits[:, 0].argmax(dim=-1)

        candidates = (logprobs.unsqueeze(1) + step_logp).flatten()
        top_logp, flat_idx = torch.topk(candidates, min(n, candidates.shape[0]))
        sources = (flat_idx // step_logp.shape[-1]).tolist()
        chars = (flat_idx % step_logp.shape[-1]).tolist()

        new_texts: list[list[int]] = []
        keep_sources: list[int] = []
        new_logprobs: list[float] = []
        for source, char, logp in zip(sources, chars, top_logp.tolist()):
            source_text = texts[source]
            if char == EOS or int(pred_classes[source]) != beam_class:
                # beam ends; the switching character is not emitted
                end_logp = logp if char == EOS else float(logprobs[source])
                finished.append(
                    Completion(char_vocab.decode(source_text), class_vocab.decode(beam_class), end_logp)
                )
                continue
            new_texts.append(source_text + [char])
            keep_sources.append(source)
            new_logprobs.append(logp)

        texts = new_texts
        if not texts:
            break
        logprobs = torch.tensor(new_logprobs, device=device)
        hidden = (state[0][:, keep_sources], state[1][:, keep_sources])

    for text, logp in zip(texts, logprobs.tolist()):  # ran into max_len
        finished.append(Completion(char_vocab.decode(text), class_vocab.decode(beam_class), logp))

    best: dict[str, Completion] = {}
    for completion in finished:
        kept = best.get(completion.text)
        if kept is None or completion.logprob > kept.logprob:
            best[completion.text] = completion
    return sorted(best.values(), key=lambda c: c.logprob, reverse=True)[:beam_width]
