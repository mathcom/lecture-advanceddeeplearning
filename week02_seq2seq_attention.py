"""Week 02 - Seq2Seq with Attention (sequence transduction).

Purpose:
    Build an encoder-decoder model with attention that learns a toy sequence
    transformation: reversing a sequence of integer tokens.

    The two attention mechanisms of week 2 are written **by hand**, so that the
    equations in the lecture notes map line by line onto the code:

      * BahdanauAttention  additive score  e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j)
                           (Bahdanau et al., 2015 - lecture section 4)
      * LuongAttention     dot / general / concat scores
                           (Luong et al., 2015 - lecture section 5)

    `nn.MultiheadAttention` is kept as a third option so the hand-written
    modules can be compared with the library one. Note that the library module
    is really the *week 3* mechanism (scaled dot-product with several heads).

    Two more options exist because the lecture argues from them:

      * attn_type="none"   the plain RNN encoder-decoder (one fixed context
                           vector c) that attention is measured against
      * wiring             "bahdanau" (query = s_{i-1}, context enters the RNN)
                           vs "luong"  (query = h_t, context enters the output)
                           plus input feeding for the Luong wiring

    Decoding offers greedy search and beam search with length normalisation,
    and the source sequence can be reversed (Sutskever et al., 2014).

Libraries:
    PyTorch (torch, torch.nn) only.

Run:
    python week02_seq2seq_attention.py

    No downloads required; data is synthetic and training is short, so it runs
    offline on CPU in about a minute.
"""

# %% [1] Imports and configuration
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 0
PAD, BOS, EOS = 0, 1, 2          # special token ids
NUM_DIGITS = 10                  # digit symbols 0..9 map to ids 3..12
VOCAB_SIZE = 3 + NUM_DIGITS      # specials + digits
MIN_LEN, MAX_LEN = 4, 10         # source sequence length range

EMB_DIM = 64
HID_DIM = 128
NUM_HEADS = 4                    # only used by attn_type="mha"
BATCH_SIZE = 64
TRAIN_STEPS = 1500
LR = 1e-3
EVAL_SAMPLES = 6                 # examples printed at the end
EVAL_BATCH = 64                  # sentences used for the accuracy number

# Which attention to build. One of:
#   "bahdanau"       hand-written additive attention        (lecture 4.3.3)
#   "luong-dot"      hand-written  score = h_t . bar_h_s    (lecture 5.3.2)
#   "luong-general"  hand-written  score = h_t^T W_a bar_h_s
#   "luong-concat"   hand-written  score = v_a^T tanh(W_a [h_t; bar_h_s])
#   "mha"            nn.MultiheadAttention (library, week-3 mechanism)
#   "none"           no attention: one fixed context vector (Cho 2014 baseline)
ATTENTION = "bahdanau"
WIRING = None                    # None -> pick the wiring the paper used
INPUT_FEEDING = False            # Luong wiring only (lecture 5.3.4)
REVERSE_SOURCE = False           # Sutskever's reversed input (lecture 2.2)

ATTENTION_TYPES = ("bahdanau", "luong-dot", "luong-general", "luong-concat",
                   "mha", "none")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# %% [2] Synthetic data (vocab, batch generator)
def digit_to_id(d: int) -> int:
    return d + 3                  # shift past PAD/BOS/EOS


def id_to_digit(i: int) -> int:
    return i - 3


def make_batch(batch_size: int, reverse_source: bool = None):
    """Sample random digit sequences; target is the reversed source.

    Returns padded tensors:
        src      [B, S]          source digits + EOS
        tgt_in   [B, T]          BOS + reversed digits (decoder input)
        tgt_out  [B, T]          reversed digits + EOS (decoder target)
        src_pad  [B, S] bool     True where src is padding

    reverse_source reverses the source tokens before padding, which is
    Sutskever's trick: it shortens the distance between the first source word
    and the first target word (lecture 2.2).
    """
    if reverse_source is None:
        reverse_source = REVERSE_SOURCE

    lengths = [random.randint(MIN_LEN, MAX_LEN) for _ in range(batch_size)]
    seqs = [[random.randint(0, NUM_DIGITS - 1) for _ in range(n)] for n in lengths]

    s_max = max(lengths) + 1     # +1 for EOS
    t_max = max(lengths) + 1     # +1 for BOS/EOS

    src = torch.full((batch_size, s_max), PAD, dtype=torch.long)
    tgt_in = torch.full((batch_size, t_max), PAD, dtype=torch.long)
    tgt_out = torch.full((batch_size, t_max), PAD, dtype=torch.long)

    for b, seq in enumerate(seqs):
        rev = seq[::-1]
        source = seq[::-1] if reverse_source else seq
        src_ids = [digit_to_id(d) for d in source] + [EOS]
        in_ids = [BOS] + [digit_to_id(d) for d in rev]
        out_ids = [digit_to_id(d) for d in rev] + [EOS]
        src[b, : len(src_ids)] = torch.tensor(src_ids)
        tgt_in[b, : len(in_ids)] = torch.tensor(in_ids)
        tgt_out[b, : len(out_ids)] = torch.tensor(out_ids)

    src_pad = src.eq(PAD)
    return (src.to(DEVICE), tgt_in.to(DEVICE),
            tgt_out.to(DEVICE), src_pad.to(DEVICE))


# %% [3] Attention modules (hand-written)
def masked_softmax(scores, mask):
    """Softmax over the last axis, with padding positions removed first.

    scores [B, S], mask [B, S] with True at padding positions. Leaving the
    padding scores in place would hand probability mass to words that are not
    there. Nothing crashes, the model just gets quietly worse - see the
    "implementation trap" note of lecture section 4.3.3.
    """
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(dim=-1)


class BahdanauAttention(nn.Module):
    """Additive attention:  e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j).

    Query is the decoder state *before* the current step. Keys and values are
    the same bidirectional annotations h_j, so the context has the annotation
    dimension 2n (lecture 4.3.2).
    """

    memory_kind = "raw"           # consumes the 2H bidirectional annotations

    def __init__(self, query_dim, mem_dim, attn_dim=None, **kwargs):
        super().__init__()
        attn_dim = attn_dim or query_dim
        self.W_a = nn.Linear(query_dim, attn_dim, bias=False)   # query  -> R^{n'}
        self.U_a = nn.Linear(mem_dim, attn_dim, bias=False)     # h_j    -> R^{n'}
        self.v_a = nn.Linear(attn_dim, 1, bias=False)           # R^{n'} -> scalar
        self.out_dim = mem_dim

    def precompute(self, memory):
        """U_a h_j does not depend on the target step, so compute it once."""
        return self.U_a(memory)                                  # [B, S, n']

    def forward(self, query, memory, mask=None, cache=None):
        Uh = self.precompute(memory) if cache is None else cache
        e = self.v_a(torch.tanh(Uh + self.W_a(query).unsqueeze(1))).squeeze(-1)
        alpha = masked_softmax(e, mask)                          # [B, S]
        context = torch.bmm(alpha.unsqueeze(1), memory).squeeze(1)
        return context, alpha.unsqueeze(1)                       # [B,1,S]


class LuongAttention(nn.Module):
    """Global attention with the three score functions of Luong et al. (2015).

        dot      score = h_t . bar_h_s           (needs equal dimensions)
        general  score = h_t^T W_a bar_h_s
        concat   score = v_a^T tanh(W_a [h_t; bar_h_s])   == Bahdanau's form

    Query is the decoder state *of the current step*.
    """

    def __init__(self, query_dim, mem_dim, score="general", attn_dim=None, **kwargs):
        super().__init__()
        if score not in ("dot", "general", "concat"):
            raise ValueError(f"score must be dot/general/concat, got {score!r}")
        self.score_type = score
        # "dot" assumes query and key already live in the same space, so it can
        # only read the projected memory (lecture 5.3.2). The other two learn
        # the mapping themselves and can read the raw 2H annotations.
        self.memory_kind = "proj" if score == "dot" else "raw"
        if score == "dot" and query_dim != mem_dim:
            raise ValueError("dot score needs query_dim == mem_dim")
        if score == "general":
            self.W_a = nn.Linear(mem_dim, query_dim, bias=False)
        elif score == "concat":
            attn_dim = attn_dim or query_dim
            self.W_a = nn.Linear(query_dim + mem_dim, attn_dim, bias=False)
            self.v_a = nn.Linear(attn_dim, 1, bias=False)
        self.out_dim = mem_dim

    def forward(self, query, memory, mask=None, cache=None):
        if self.score_type == "dot":
            e = torch.bmm(memory, query.unsqueeze(-1)).squeeze(-1)
        elif self.score_type == "general":
            e = torch.bmm(self.W_a(memory), query.unsqueeze(-1)).squeeze(-1)
        else:                                    # concat == additive
            q = query.unsqueeze(1).expand(-1, memory.size(1), -1)
            e = self.v_a(torch.tanh(self.W_a(torch.cat([q, memory], dim=-1)))).squeeze(-1)
        alpha = masked_softmax(e, mask)
        context = torch.bmm(alpha.unsqueeze(1), memory).squeeze(1)
        return context, alpha.unsqueeze(1)


class LibraryMultiHeadAttention(nn.Module):
    """nn.MultiheadAttention wrapped in the same interface.

    This is scaled dot-product attention with several heads - the week-3
    mechanism. It is here so the notebook can compare it against the
    hand-written modules under exactly the same wiring.
    """

    memory_kind = "proj"

    def __init__(self, query_dim, mem_dim, heads=NUM_HEADS, **kwargs):
        super().__init__()
        if query_dim != mem_dim:
            raise ValueError("nn.MultiheadAttention needs query_dim == mem_dim")
        self.attn = nn.MultiheadAttention(query_dim, heads, batch_first=True)
        self.heads = heads
        self.out_dim = query_dim

    def forward(self, query, memory, mask=None, cache=None):
        # average_attn_weights=False keeps one distribution per head. The
        # default (True) returns the head average, which is NOT the alpha_ij of
        # week 2 and quietly blurs the alignment picture.
        context, weights = self.attn(query.unsqueeze(1), memory, memory,
                                     key_padding_mask=mask,
                                     average_attn_weights=False)
        return context.squeeze(1), weights.squeeze(2)            # [B, heads, S]


class NoAttention(nn.Module):
    """The RNN encoder-decoder baseline: one fixed context vector c.

    mode="last" takes the last non-padding annotation, i.e. c = h_{Tx}, which is
    what Cho (2014) and Sutskever (2014) use. mode="mean" averages all
    annotations, i.e. attention with frozen uniform weights.
    """

    memory_kind = "raw"

    def __init__(self, query_dim, mem_dim, mode="last", **kwargs):
        super().__init__()
        self.mode = mode
        self.out_dim = mem_dim

    def forward(self, query, memory, mask=None, cache=None):
        if mask is None:
            keep = torch.ones(memory.shape[:2], device=memory.device)
        else:
            keep = (~mask).float()
        if self.mode == "last":
            last = keep.sum(1).long().clamp(min=1) - 1           # index of h_{Tx}
            context = memory[torch.arange(memory.size(0), device=memory.device), last]
            alpha = F.one_hot(last, memory.size(1)).float()
        else:
            alpha = keep / keep.sum(1, keepdim=True)
            context = torch.bmm(alpha.unsqueeze(1), memory).squeeze(1)
        return context, alpha.unsqueeze(1).detach()


DEFAULT_WIRING = {
    "bahdanau": "bahdanau", "mha": "bahdanau", "none": "bahdanau",
    "luong-dot": "luong", "luong-general": "luong", "luong-concat": "luong",
}


def build_attention(attn_type, query_dim, mem_dim, heads=NUM_HEADS):
    if attn_type == "bahdanau":
        return BahdanauAttention(query_dim, mem_dim)
    if attn_type.startswith("luong-"):
        return LuongAttention(query_dim, mem_dim, score=attn_type.split("-", 1)[1])
    if attn_type == "mha":
        return LibraryMultiHeadAttention(query_dim, mem_dim, heads=heads)
    if attn_type == "none":
        return NoAttention(query_dim, mem_dim)
    raise ValueError(f"attn_type must be one of {ATTENTION_TYPES}, got {attn_type!r}")


def needs_projected_memory(attn_type, heads=NUM_HEADS):
    """True when the attention can only read same-dimension keys/values."""
    probe = {"bahdanau": BahdanauAttention, "mha": LibraryMultiHeadAttention,
             "none": NoAttention}.get(attn_type)
    if probe is not None:
        return getattr(probe, "memory_kind", "raw") == "proj"
    return attn_type == "luong-dot"


# %% [4] Model (bidirectional GRU encoder, attentional GRU decoder)
class Encoder(nn.Module):
    """Bidirectional GRU. h_j = [forward_j ; backward_j] are the annotations.

    NOTE on nn.GRU: PyTorch applies the reset gate *after* the hidden matrix
    multiply, r * (W_hn h + b_hn), while Cho's paper applies it before,
    U (r * h). With biases the two are not the same function (lecture 1.3.3).
    """

    def __init__(self, vocab, emb, hid, project_memory=False):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hid, hid)                    # s_0
        self.mem_proj = nn.Linear(2 * hid, hid) if project_memory else None
        self.out_dim = hid if project_memory else 2 * hid

    def forward(self, src):
        memory, h = self.rnn(self.emb(src))                      # [B,S,2H], [2,B,H]
        s0 = torch.tanh(self.bridge(torch.cat([h[0], h[1]], dim=-1)))
        if self.mem_proj is not None:
            memory = self.mem_proj(memory)
        return memory, s0.unsqueeze(0)                           # decoder is 1 layer


class Decoder(nn.Module):
    """Single-step GRU decoder with a pluggable attention module.

    Two wirings, exactly as the two papers describe them:

      bahdanau:  s_{i-1} -> alpha_i -> c_i -> s_i,  p(y_i) = g(y_{i-1}, s_i, c_i)
      luong:     h_t     -> a_t     -> c_t -> ~h_t = tanh(W_c [c_t; h_t]),
                 p(y_t) = softmax(W_s ~h_t)     (+ optional input feeding)
    """

    def __init__(self, vocab, emb, hid, mem_dim, attn_type="bahdanau",
                 wiring=None, heads=NUM_HEADS, input_feeding=False):
        super().__init__()
        self.attn_type = attn_type
        self.wiring = wiring or DEFAULT_WIRING[attn_type]
        if self.wiring not in ("bahdanau", "luong"):
            raise ValueError(f"wiring must be bahdanau/luong, got {self.wiring!r}")
        self.hid = hid
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.attn = build_attention(attn_type, hid, mem_dim, heads)
        ctx_dim = self.attn.out_dim
        self.ctx_dim = ctx_dim
        self.input_feeding = bool(input_feeding) and self.wiring == "luong"

        if self.wiring == "bahdanau":
            self.rnn = nn.GRU(emb + ctx_dim, hid, batch_first=True)
            self.out = nn.Linear(hid + ctx_dim + emb, vocab)     # g(y_{i-1}, s_i, c_i)
        else:
            self.rnn = nn.GRU(emb + (hid if self.input_feeding else 0), hid,
                              batch_first=True)
            self.W_c = nn.Linear(ctx_dim + hid, hid, bias=False)  # ~h_t
            self.out = nn.Linear(hid, vocab)                      # W_s

    # -- one decoding step ------------------------------------------------
    def step(self, y_emb, h, memory, src_pad, cache=None, prev_att=None):
        """y_emb [B, emb] embedding of the previous token. Returns
        (logits [B,V], new h, alpha [B,heads,S], attentional state or None)."""
        if self.wiring == "bahdanau":
            query = h[-1]                                        # s_{i-1}
            context, alpha = self.attn(query, memory, src_pad, cache)
            out, h = self.rnn(torch.cat([y_emb, context], -1).unsqueeze(1), h)
            s_i = out.squeeze(1)
            logits = self.out(torch.cat([s_i, context, y_emb], -1))
            return logits, h, alpha, None

        rnn_in = y_emb if not self.input_feeding else torch.cat([y_emb, prev_att], -1)
        out, h = self.rnn(rnn_in.unsqueeze(1), h)
        h_t = out.squeeze(1)                                     # query = h_t
        context, alpha = self.attn(h_t, memory, src_pad, cache)
        att_h = torch.tanh(self.W_c(torch.cat([context, h_t], -1)))
        return self.out(att_h), h, alpha, att_h

    def init_att(self, batch, device):
        return torch.zeros(batch, self.hid, device=device)

    def cache_for(self, memory):
        return self.attn.precompute(memory) if hasattr(self.attn, "precompute") else None

    # -- teacher forcing over a whole target sequence ---------------------
    def forward(self, tgt_in, h, memory, src_pad):
        emb = self.emb(tgt_in)
        cache = self.cache_for(memory)
        prev_att = self.init_att(emb.size(0), emb.device)
        logits, alphas = [], []
        for t in range(emb.size(1)):
            lg, h, alpha, att = self.step(emb[:, t], h, memory, src_pad, cache, prev_att)
            if att is not None:
                prev_att = att
            logits.append(lg)
            alphas.append(alpha)
        return torch.stack(logits, 1), h, torch.stack(alphas, 1)  # [B,T,V], [B,T,heads,S]


class Seq2Seq(nn.Module):
    def __init__(self, attn_type=None, wiring=None, input_feeding=None, heads=None):
        super().__init__()
        attn_type = ATTENTION if attn_type is None else attn_type
        wiring = WIRING if wiring is None else wiring
        input_feeding = INPUT_FEEDING if input_feeding is None else input_feeding
        heads = NUM_HEADS if heads is None else heads
        if attn_type not in ATTENTION_TYPES:
            raise ValueError(f"attn_type must be one of {ATTENTION_TYPES}")

        self.attn_type = attn_type
        self.encoder = Encoder(VOCAB_SIZE, EMB_DIM, HID_DIM,
                               project_memory=needs_projected_memory(attn_type))
        self.decoder = Decoder(VOCAB_SIZE, EMB_DIM, HID_DIM, self.encoder.out_dim,
                               attn_type=attn_type, wiring=wiring, heads=heads,
                               input_feeding=input_feeding)

    def forward(self, src, tgt_in, src_pad):
        memory, h = self.encoder(src)
        logits, _, _ = self.decoder(tgt_in, h, memory, src_pad)
        return logits

    def forward_with_attention(self, src, tgt_in, src_pad):
        """Same as forward(), but also returns alpha [B, T, heads, S]."""
        memory, h = self.encoder(src)
        logits, _, alphas = self.decoder(tgt_in, h, memory, src_pad)
        return logits, alphas

    def describe(self):
        return (f"attention={self.attn_type}  wiring={self.decoder.wiring}  "
                f"input_feeding={self.decoder.input_feeding}  "
                f"memory_dim={self.encoder.out_dim}  context_dim={self.decoder.ctx_dim}  "
                f"params={sum(p.numel() for p in self.parameters()):,}")


# %% [5] Train utils (loss with padding ignore, optimizer)
def build_optim(model, lr=None):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)   # skip pad positions
    optimizer = torch.optim.Adam(model.parameters(), lr=LR if lr is None else lr)
    return criterion, optimizer


# %% [6] Training loop
def train(model, criterion, optimizer, steps=None, log_every=250, verbose=True):
    steps = TRAIN_STEPS if steps is None else steps
    model.train()
    for step in range(1, steps + 1):
        src, tgt_in, tgt_out, src_pad = make_batch(BATCH_SIZE)
        logits = model(src, tgt_in, src_pad)
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # lecture 1 section 2.3
        optimizer.step()

        if verbose and (step % log_every == 0 or step == 1):
            print(f"step {step:4d}/{steps}  loss {loss.item():.4f}")
    return model


# %% [7] Decoding: greedy and beam search
@torch.no_grad()
def greedy_decode(model, src, src_pad, max_len, return_attention=False):
    """Autoregressively decode one greedy token at a time.

    Returns ids [B, L], or (ids, alpha [B, L, heads, S]) when asked.
    """
    model.eval()
    memory, h = model.encoder(src)
    dec = model.decoder
    cache = dec.cache_for(memory)

    batch = src.size(0)
    cur = torch.full((batch,), BOS, dtype=torch.long, device=src.device)
    prev_att = dec.init_att(batch, src.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
    outputs, alphas = [], []

    for _ in range(max_len):
        logits, h, alpha, att = dec.step(dec.emb(cur), h, memory, src_pad, cache, prev_att)
        if att is not None:
            prev_att = att
        cur = logits.argmax(-1)                              # [B]
        outputs.append(cur)
        alphas.append(alpha)
        finished |= cur.eq(EOS)
        if bool(finished.all()):
            break

    ids = torch.stack(outputs, dim=1)                        # [B, L]
    if not return_attention:
        return ids
    return ids, torch.stack(alphas, dim=1)                   # [B, L, heads, S]


@torch.no_grad()
def beam_decode(model, src, src_pad, max_len, beam=3, length_norm=0.0):
    """Beam search for ONE source sentence. src [1, S], src_pad [1, S].

    length_norm is the exponent in  score / len**length_norm :
        0.0  raw sum of log probabilities - biased towards short outputs,
             which is why "a wider beam made BLEU worse" (lecture 2.3)
        1.0  mean log probability per token (length normalisation)
    Returns a list of token ids without the trailing EOS.
    """
    model.eval()
    if src.size(0) != 1:
        raise ValueError("beam_decode expects a single sentence; loop over the batch")
    dec = model.decoder
    memory, h = model.encoder(src)
    memory = memory.expand(beam, -1, -1).contiguous()
    mask = src_pad.expand(beam, -1).contiguous()
    h = h.expand(-1, beam, -1).contiguous()
    cache = dec.cache_for(memory)
    prev_att = dec.init_att(beam, src.device)

    cur = torch.full((beam,), BOS, dtype=torch.long, device=src.device)
    # Only the first beam is alive at step 1 (all beams are identical there).
    scores = torch.full((beam,), float("-inf"), device=src.device)
    scores[0] = 0.0
    seqs = [[] for _ in range(beam)]
    completed = []

    for _ in range(max_len):
        logits, h_new, _, att = dec.step(dec.emb(cur), h, memory, mask, cache, prev_att)
        logp = F.log_softmax(logits.float(), dim=-1)             # [beam, V]
        total = scores.unsqueeze(1) + logp
        flat = total.reshape(-1)
        top_scores, top_idx = flat.topk(beam)
        beam_idx = torch.div(top_idx, logits.size(-1), rounding_mode="floor")
        token = top_idx % logits.size(-1)

        h = h_new[:, beam_idx, :].contiguous()
        prev_att = att[beam_idx] if att is not None else prev_att
        seqs = [seqs[b] + [int(t)] for b, t in zip(beam_idx.tolist(), token.tolist())]
        scores = top_scores.clone()
        cur = token

        for b in range(beam):
            if token[b].item() == EOS:
                completed.append((scores[b].item(), seqs[b][:-1]))
                scores[b] = float("-inf")
        if len(completed) >= beam or bool(torch.isinf(scores).all()):
            break

    if not completed:                                            # hit max_len
        order = scores.argsort(descending=True)
        completed = [(scores[i].item(), seqs[i]) for i in order.tolist()
                     if not math.isinf(scores[i].item())]
        if not completed:
            return []
    best = max(completed, key=lambda sl: sl[0] / max(len(sl[1]), 1) ** length_norm)
    return best[1]


def ids_to_digits(row) -> list:
    """Trim a decoded id row at the first EOS and map back to digits."""
    out = []
    for i in (row.tolist() if torch.is_tensor(row) else row):
        if i == EOS:
            break
        if i >= 3:
            out.append(id_to_digit(i))
    return out


# %% [8] Evaluation
def evaluate(model, num_samples, decoder="greedy", beam=3, length_norm=0.0):
    src, _, tgt_out, src_pad = make_batch(num_samples)
    if decoder == "greedy":
        pred = greedy_decode(model, src, src_pad, max_len=MAX_LEN + 1)
        preds = [ids_to_digits(pred[b]) for b in range(num_samples)]
    else:
        preds = [ids_to_digits(beam_decode(model, src[b:b + 1], src_pad[b:b + 1],
                                           max_len=MAX_LEN + 1, beam=beam,
                                           length_norm=length_norm))
                 for b in range(num_samples)]

    exact, rows = 0, []
    for b in range(num_samples):
        src_d = ids_to_digits(src[b])
        gold_d = ids_to_digits(tgt_out[b])
        exact += int(preds[b] == gold_d)
        rows.append((src_d, gold_d, preds[b]))
    return exact / num_samples, rows


# %% [9] main()
def main():
    set_seed(SEED)
    print(f"device: {DEVICE}")

    model = Seq2Seq().to(DEVICE)
    print(model.describe())
    criterion, optimizer = build_optim(model)
    train(model, criterion, optimizer)

    acc, rows = evaluate(model, EVAL_BATCH)
    print("\nSample reversals (greedy decode):")
    for src_d, gold_d, pred_d in rows[:EVAL_SAMPLES]:
        mark = "OK" if pred_d == gold_d else "XX"
        print(f"  [{mark}] in={src_d}  gold={gold_d}  pred={pred_d}")
    print(f"\nexact-match accuracy (greedy): {acc * 100:.1f}% ({EVAL_BATCH} samples)")

    beam_acc, _ = evaluate(model, EVAL_BATCH // 2, decoder="beam", beam=3, length_norm=1.0)
    print(f"exact-match accuracy (beam=3, length_norm=1.0): {beam_acc * 100:.1f}% "
          f"({EVAL_BATCH // 2} samples)")

    # One alignment row: which source position did the first output token read?
    src, _, _, src_pad = make_batch(1)
    _, alpha = greedy_decode(model, src, src_pad, max_len=MAX_LEN + 1,
                             return_attention=True)
    weights = alpha[0, :, 0, :]                     # first head
    print("\nattention of the first 3 output steps (rows) over the source (cols):")
    for t in range(min(3, weights.size(0))):
        print("  " + " ".join(f"{w:.2f}" for w in weights[t].tolist()))
    print("  source digits:", ids_to_digits(src[0]),
          "(reversed input)" if REVERSE_SOURCE else "")


if __name__ == "__main__":
    main()
