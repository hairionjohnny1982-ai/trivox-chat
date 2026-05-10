"""TriVox encoder — auto-detects v2 (768d) or v3 (384d) from checkpoint."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from tokenizers import Tokenizer


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ffn_fc1 = nn.Linear(d_model, d_ff)
        self.ffn_fc2 = nn.Linear(d_ff, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x, mask=None):
        h = self.norm1(x)
        B, L, D = h.shape
        qkv = self.attn_qkv(h).reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.d_head ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.attn_out(out)
        h = self.norm2(x)
        x = x + self.ffn_fc2(F.gelu(self.ffn_fc1(h)))
        return x


class TriVoxModel(nn.Module):
    def __init__(self, d_model=512, n_layers=6, n_heads=8, d_ff=2048,
                 vocab_fr=50000, vocab_en=50000, vocab_code=32000,
                 max_seq_len=256, d_out=768):
        super().__init__()
        self.embed_fr = nn.Embedding(vocab_fr, d_model)
        self.embed_en = nn.Embedding(vocab_en, d_model)
        self.embed_code = nn.Embedding(vocab_code, d_model)
        self.lang_embed = nn.Embedding(3, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(nn.Linear(d_model, d_out), nn.LayerNorm(d_out))

    def forward(self, ids, lang_id=0, mask=None):
        B, L = ids.shape
        if lang_id == 0:
            x = self.embed_fr(ids)
        elif lang_id == 1:
            x = self.embed_en(ids)
        else:
            x = self.embed_code(ids)
        positions = torch.arange(L, device=ids.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)
        lang_t = torch.full((B,), lang_id, device=ids.device, dtype=torch.long)
        x = x + self.lang_embed(lang_t).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, mask)
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        return self.proj(pooled)


def _detect_architecture(state_dict: dict) -> dict:
    """Auto-detect model architecture from checkpoint weights."""
    # Check embed_fr size to get d_model
    if "embed_fr.weight" in state_dict:
        vocab_fr, d_model = state_dict["embed_fr.weight"].shape
    else:
        vocab_fr, d_model = 50000, 512

    # Check projection output dim
    for key in ["proj.0.weight", "proj.weight"]:
        if key in state_dict:
            d_out = state_dict[key].shape[0]
            break
    else:
        d_out = 768

    # Count layers
    n_layers = 0
    for key in state_dict:
        m = __import__("re").match(r"layers\.(\d+)\.", key)
        if m:
            n_layers = max(n_layers, int(m.group(1)) + 1)
    if n_layers == 0:
        n_layers = 6

    # Detect n_heads from qkv weight shape
    for key in state_dict:
        if "attn_qkv.weight" in key or "attn.qkv.weight" in key:
            qkv_dim = state_dict[key].shape[0]
            # qkv_dim = 3 * d_model, n_heads = d_model / d_head
            # Try common head sizes: 32, 64
            for d_head in [64, 32]:
                if d_model % d_head == 0:
                    n_heads = d_model // d_head
                    break
            else:
                n_heads = 8
            break
    else:
        n_heads = d_model // 64 if d_model >= 256 else 4

    # Detect d_ff from ffn_fc1
    d_ff = d_model * 4
    for key in state_dict:
        if "ffn_fc1.weight" in key or "ffn.fc1.weight" in key:
            d_ff = state_dict[key].shape[0]
            break

    # Detect vocab sizes
    vocab_en = state_dict.get("embed_en.weight", torch.zeros(50000, d_model)).shape[0]
    vocab_code = state_dict.get("embed_code.weight", torch.zeros(32000, d_model)).shape[0]

    arch = {
        "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "d_ff": d_ff, "vocab_fr": vocab_fr, "vocab_en": vocab_en,
        "vocab_code": vocab_code, "d_out": d_out,
    }
    return arch


class Encoder:
    def __init__(self, model_path: str, tok_fr: str, tok_en: str, tok_code: str,
                 max_seq_len: int = 256):
        self.device = torch.device("cpu")
        self.max_seq_len = max_seq_len

        print(f"[Encoder] Loading model from {model_path}...")
        raw = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "model" in raw:
            sd = raw["model"]
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            sd = raw["model_state_dict"]
        else:
            sd = raw

        # Map checkpoint keys
        mapped = {}
        for k, v in sd.items():
            new_k = k.replace(".attn.qkv.", ".attn_qkv.").replace(".attn.out.", ".attn_out.")
            new_k = new_k.replace(".ffn.fc1.", ".ffn_fc1.").replace(".ffn.fc2.", ".ffn_fc2.")
            mapped[new_k] = v

        # Auto-detect architecture
        arch = _detect_architecture(mapped)
        print(f"[Encoder] Detected: d_model={arch['d_model']}, d_out={arch['d_out']}, "
              f"layers={arch['n_layers']}, heads={arch['n_heads']}")
        self.embed_dim = arch["d_out"]

        self.model = TriVoxModel(
            d_model=arch["d_model"], n_layers=arch["n_layers"],
            n_heads=arch["n_heads"], d_ff=arch["d_ff"],
            vocab_fr=arch["vocab_fr"], vocab_en=arch["vocab_en"],
            vocab_code=arch["vocab_code"],
            max_seq_len=max_seq_len, d_out=arch["d_out"],
        )
        missing, unexpected = self.model.load_state_dict(mapped, strict=False)
        if missing:
            print(f"[Encoder] Warning: {len(missing)} missing keys")
        self.model.eval()
        print(f"[Encoder] Model loaded ({arch['d_out']}d)")

        self.tok_fr = spm.SentencePieceProcessor(model_file=tok_fr)
        self.tok_en = spm.SentencePieceProcessor(model_file=tok_en)
        self.tok_code = Tokenizer.from_file(tok_code)
        print(f"[Encoder] Tokenizers loaded")

    def detect_lang(self, text: str) -> int:
        t = text[:500].lower()
        fr = sum(1 for w in ["le ", "la ", "les ", "de ", "du ", "est ", "une ",
                             "je ", "tu ", "nous ", "mon ", "dans "]
                 if w in t)
        code = sum(1 for w in ["def ", "class ", "import ", "return ", "function ",
                               "const ", "var ", "if (", "for (", "while (", "self."]
                  if w in t)
        if code >= 2:
            return 2
        if fr >= 2:
            return 0
        return 1

    @torch.no_grad()
    def encode(self, text: str) -> list[float]:
        lang = self.detect_lang(text)
        if lang == 0:
            ids = self.tok_fr.encode(str(text), out_type=int)[:self.max_seq_len]
        elif lang == 1:
            ids = self.tok_en.encode(str(text), out_type=int)[:self.max_seq_len]
        else:
            ids = self.tok_code.encode(str(text)).ids[:self.max_seq_len]
        if not ids:
            return [0.0] * self.embed_dim
        pad = self.max_seq_len - len(ids)
        ids_t = torch.tensor([ids + [0] * pad], dtype=torch.long, device=self.device)
        mask_t = torch.tensor([[1] * len(ids) + [0] * pad], dtype=torch.long, device=self.device)
        emb = self.model(ids_t, lang, mask_t).squeeze(0)
        emb = emb / emb.norm().clamp(min=1e-8)
        return emb.numpy().tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]
