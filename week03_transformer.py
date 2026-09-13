"""Week 03 - Transformer with PyTorch built-in modules.

A seq2seq Transformer (nn.Transformer) trained on a synthetic copy/sort task.
We rely on torch's own attention, masking helpers, and encoder/decoder stacks
instead of re-implementing them. The script generates synthetic data, trains
briefly, then runs greedy auto-regressive decoding and reports exact-match
accuracy. Runnable fully offline on CPU.
"""

# %% Imports & config
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Special token ids. Vocabulary = {PAD, BOS, EOS} + content symbols [3 .. VOCAB-1].
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
NUM_SPECIAL = 3
VOCAB_SIZE = 20          # 3 special + 17 content symbols
MAX_CONTENT_LEN = 10     # content length per sequence (excludes special tokens)
TASK = "sort"            # one of: "copy", "reverse", "sort"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# %% Synthetic data
def transform(symbols: list[int]) -> list[int]:
    """Map an input symbol sequence to the target for the chosen task."""
    if TASK == "copy":
        return list(symbols)
    if TASK == "reverse":
        return list(reversed(symbols))
    if TASK == "sort":
        return sorted(symbols)
    raise ValueError(f"unknown task: {TASK}")


def make_pair() -> tuple[list[int], list[int]]:
    """Build one (src, tgt) pair with random length and content symbols."""
    length = random.randint(MAX_CONTENT_LEN // 2, MAX_CONTENT_LEN)
    symbols = [random.randint(NUM_SPECIAL, VOCAB_SIZE - 1) for _ in range(length)]
    src = symbols + [EOS_ID]
    tgt = [BOS_ID] + transform(symbols) + [EOS_ID]
    return src, tgt


def pad_to(seq: list[int], length: int) -> list[int]:
    return seq + [PAD_ID] * (length - len(seq))


def build_dataset(n: int) -> TensorDataset:
    """Generate n pairs and right-pad them to fixed tensor shapes."""
    pairs = [make_pair() for _ in range(n)]
    src_len = max(len(s) for s, _ in pairs)
    tgt_len = max(len(t) for _, t in pairs)
    src = torch.tensor([pad_to(s, src_len) for s, _ in pairs], dtype=torch.long)
    tgt = torch.tensor([pad_to(t, tgt_len) for _, t in pairs], dtype=torch.long)
    return TensorDataset(src, tgt)


# %% Positional encoding + Transformer model (nn.Transformer wrapper)
class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    """Thin wrapper around nn.Transformer for token-to-token seq2seq."""

    def __init__(self, vocab: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True,
        )
        self.generator = nn.Linear(d_model, vocab)

    def _embed(self, tokens: torch.Tensor, emb: nn.Embedding) -> torch.Tensor:
        # Scale embeddings as in the original paper, then add positions.
        return self.pos(emb(tokens) * math.sqrt(self.d_model))

    def encode(self, src: torch.Tensor, src_pad: torch.Tensor) -> torch.Tensor:
        return self.transformer.encoder(
            self._embed(src, self.src_emb), src_key_padding_mask=src_pad)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor, tgt_pad: torch.Tensor,
               mem_pad: torch.Tensor) -> torch.Tensor:
        out = self.transformer.decoder(
            self._embed(tgt, self.tgt_emb), memory,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=mem_pad)
        return self.generator(out)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                tgt_mask: torch.Tensor, src_pad: torch.Tensor,
                tgt_pad: torch.Tensor) -> torch.Tensor:
        memory = self.encode(src, src_pad)
        return self.decode(tgt, memory, tgt_mask, tgt_pad, src_pad)


# %% Masks & train utils
def padding_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Boolean mask: True where the position is PAD (ignored by attention)."""
    return tokens == PAD_ID


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Look-ahead mask via torch helper (-inf above the diagonal)."""
    return nn.Transformer.generate_square_subsequent_mask(size, device=device)


def run_epoch(model, loader, loss_fn, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]  # teacher forcing shift

        mask = causal_mask(tgt_in.size(1), DEVICE)
        logits = model(src, tgt_in, mask, padding_mask(src), padding_mask(tgt_in))
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total += loss.item() * src.size(0)
        count += src.size(0)
    return total / count


# %% Greedy auto-regressive decode
@torch.no_grad()
def greedy_decode(model, src: torch.Tensor, max_len: int) -> torch.Tensor:
    """Decode one batch left-to-right, picking the argmax token each step."""
    model.eval()
    src = src.to(DEVICE)
    src_pad = padding_mask(src)
    memory = model.encode(src, src_pad)

    ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long, device=DEVICE)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=DEVICE)
    for _ in range(max_len - 1):
        mask = causal_mask(ys.size(1), DEVICE)
        logits = model.decode(ys, memory, mask, padding_mask(ys), src_pad)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        nxt[done] = PAD_ID  # keep finished rows padded
        ys = torch.cat([ys, nxt], dim=1)
        done |= nxt.squeeze(1) == EOS_ID
        if bool(done.all()):
            break
    return ys


def strip_specials(seq: list[int]) -> list[int]:
    """Drop BOS/EOS/PAD, stopping at the first EOS."""
    out = []
    for t in seq:
        if t == EOS_ID:
            break
        if t not in (BOS_ID, PAD_ID):
            out.append(t)
    return out


def evaluate(model, loader, max_len: int, show: int = 5) -> float:
    correct, total, shown = 0, 0, 0
    for src, tgt in loader:
        pred = greedy_decode(model, src, max_len).cpu().tolist()
        for i in range(len(src)):
            ref = strip_specials(tgt[i].tolist())
            hyp = strip_specials(pred[i])
            correct += int(ref == hyp)
            total += 1
            if shown < show:
                src_sym = strip_specials(src[i].tolist())
                print(f"  src={src_sym} -> pred={hyp} | gold={ref} "
                      f"[{'OK' if ref == hyp else 'X'}]")
                shown += 1
    return correct / total


# %% main()
def main() -> None:
    set_seed(SEED)
    print(f"device={DEVICE} task={TASK} vocab={VOCAB_SIZE}")

    train_ds = build_dataset(4000)
    test_ds = build_dataset(400)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = Seq2SeqTransformer(VOCAB_SIZE).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98))

    epochs = 10
    for ep in range(1, epochs + 1):
        loss = run_epoch(model, train_loader, loss_fn, optimizer)
        print(f"epoch {ep:2d}/{epochs}  train_loss={loss:.4f}")

    # Decode budget: target content + BOS/EOS, with a small safety margin.
    max_len = MAX_CONTENT_LEN + 4
    print("\nSample predictions:")
    acc = evaluate(model, test_loader, max_len, show=5)
    print(f"\nExact-match accuracy: {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
