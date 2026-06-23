"""
model_manager.py
================
Load mô hình embedding đúng với Database Qdrant:
  - Dense: Qwen/Qwen3-Embedding-8B + LoRA adapter từ HuggingFace
             (ngovanphuoc2006/Legal-embedding)
  - Sparse: BM25 tokenize + blake2b hash (giống hệt file tạo dataset)
"""

import hashlib
import logging
import os
import re
import unicodedata
from collections import Counter

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig

# Patch os.symlink for Windows to bypass WinError 1314 when Developer Mode is off
if os.name == "nt":
    _original_symlink = os.symlink

    def _patched_symlink(src, dst, target_is_directory=False, **kwargs):
        try:
            _original_symlink(src, dst, target_is_directory=target_is_directory, **kwargs)
        except OSError as e:
            if getattr(e, "winerror", None) == 1314:
                import shutil
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            else:
                raise e

    os.symlink = _patched_symlink

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
HF_ADAPTER_REPO = "ngovanphuoc2006/Legal-embedding"
MAX_LENGTH = 512
BATCH_SIZE = 16

# Regex dùng để tokenize — giống hệt notebook tạo database
LEGAL_CODE_RE = re.compile(r"\b\d{1,4}/\d{4}/[a-zA-ZĐđ\-]+\b")
TOKEN_RE = re.compile(r"[a-zA-ZÀ-ỹĐđ0-9]+(?:[\-/][a-zA-ZÀ-ỹĐđ0-9]+)*")


# ── Helpers (giống hệt notebook) ────────────────────────────────────────────

def _normalize_text_basic(text: str) -> str:
    """Lower-case, NFC normalize, collapse whitespace."""
    text = str(text or "").lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_vn_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize y hệt notebook tạo database (legal codes + unicode tokens + accentless shadow)."""
    text = _normalize_text_basic(text)
    tokens = []
    for m in LEGAL_CODE_RE.finditer(text):
        tokens.append(m.group(0))
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0).strip("-/_")
        if len(tok) < 2:
            continue
        tokens.append(tok)
        de = _strip_vn_accents(tok)
        if de != tok and len(de) >= 3:
            tokens.append("noaccent:" + de)
    return tokens


def token_to_sparse_id(token: str) -> int:
    """Blake2b hash → unsigned 31-bit int (positive), giống hệt notebook."""
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    val = int.from_bytes(h, "little") & 0x7FFFFFFF
    return val if val > 0 else 1


def build_query_sparse_vector(query: str) -> dict:
    """
    Chuyển câu hỏi thành Sparse Vector để gửi lên Qdrant.
    Dùng term-frequency làm score (giản dị, tương thích BM25 index).
    """
    tokens = bm25_tokenize(query)
    if not tokens:
        return {"indices": [], "values": []}

    tf: Counter = Counter(token_to_sparse_id(t) for t in tokens)
    # Normalize TF về [0, 1]
    max_tf = max(tf.values())
    items = sorted(tf.items(), key=lambda x: x[0])  # sort by index
    return {
        "indices": [int(k) for k, _ in items],
        "values":  [round(float(v / max_tf), 6) for _, v in items],
    }


# ── Model Manager ────────────────────────────────────────────────────────────

class LegalEmbeddingModel:
    """
    Wrapper cho Qwen3-Embedding-8B + LoRA adapter fine-tune pháp lý.
    Dùng last-token pooling (giống encode trong notebook tạo data).
    """

    def __init__(self):
        logger.info("Loading PeftConfig from HuggingFace: %s", HF_ADAPTER_REPO)
        config = PeftConfig.from_pretrained(HF_ADAPTER_REPO)
        base_model_id = config.base_model_name_or_path
        logger.info("Base model: %s", base_model_id)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger.info("Loading base model in %s on %s ...", dtype, device_map)
        base = AutoModel.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        base.config.use_cache = False

        logger.info("Loading LoRA adapter from: %s", HF_ADAPTER_REPO)
        self.model = PeftModel.from_pretrained(base, HF_ADAPTER_REPO)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        logger.info("Loading tokenizer from: %s", base_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = next(self.model.parameters()).device
        logger.info("LegalEmbeddingModel ready on device: %s", self.device)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode danh sách text → dense numpy array [N, 4096]."""
        all_vecs = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start: start + BATCH_SIZE]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)

            # last-token pooling (giống notebook tạo data)
            hidden = outputs.last_hidden_state  # [B, T, D]
            attention_mask = encoded["attention_mask"]  # [B, T]
            # left-padding: last token in sequence is last non-pad
            seq_lens = attention_mask.sum(dim=1) - 1  # [B]
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            pooled = hidden[batch_idx, seq_lens]  # [B, D]

            # L2-normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            all_vecs.append(pooled.float().cpu().numpy())

        return np.vstack(all_vecs).astype("float32")


class ModelManager:
    _embed_model: LegalEmbeddingModel | None = None

    @classmethod
    def get_embed_model(cls) -> LegalEmbeddingModel:
        """Singleton — chỉ load model một lần duy nhất."""
        if cls._embed_model is None:
            logger.info("Loading LegalEmbeddingModel for the first time...")
            cls._embed_model = LegalEmbeddingModel()
            logger.info("LegalEmbeddingModel loaded successfully!")
        return cls._embed_model
