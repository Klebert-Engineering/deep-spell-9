"""Token spelling encoder (successor of ``DSVariationalLstmAutoEncoder``).

Maps a single token string to a unit-length embedding. Trained contrastively
on (corrupted, clean) pairs so that misspellings land near their correct
form; nearest-neighbour search over the corpus vocabulary then yields
spelling corrections beyond fixed edit-distance limits.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..charset import CharVocab


class TokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 48,
        hidden: int = 128,
        layers: int = 1,
        out_dim: int = 32,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True
        )
        self.projection = nn.Linear(2 * hidden, out_dim)

    def forward(self, char_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``char_ids`` [B, T] -> L2-normalized embeddings [B, out_dim]."""
        embedded = self.embedding(char_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True, total_length=char_ids.shape[1]
        )
        mask = (char_ids != 0).unsqueeze(-1).to(unpacked.dtype)
        pooled = (unpacked * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return nn.functional.normalize(self.projection(pooled), dim=-1)

    @torch.inference_mode()
    def encode_strings(
        self, strings: list[str], char_vocab: CharVocab, batch_size: int = 512
    ) -> np.ndarray:
        """Encode normalized strings to a [N, out_dim] float32 matrix."""
        device = next(self.parameters()).device
        chunks: list[np.ndarray] = []
        for start in range(0, len(strings), batch_size):
            batch = strings[start : start + batch_size]
            encoded = [char_vocab.encode(s) or [0] for s in batch]
            lengths = torch.tensor([len(e) for e in encoded])
            ids = torch.zeros(len(batch), int(lengths.max()), dtype=torch.long, device=device)
            for row, ids_row in enumerate(encoded):
                ids[row, : len(ids_row)] = torch.tensor(ids_row, device=device)
            chunks.append(self(ids, lengths).cpu().numpy())
        if not chunks:
            return np.empty((0, self.projection.out_features), dtype=np.float32)
        return np.concatenate(chunks, axis=0).astype(np.float32)
