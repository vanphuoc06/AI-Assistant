"""
search_engine.py
================
Hybrid RAG Retriever sử dụng Qdrant Named Vectors:
  - dense        : Qwen3-Embedding-8B (4096 chiều, Cosine)
  - text_sparse  : BM25 sparse vector (blake2b hash IDs)

Luồng:
  1. Query Expansion  — tạo nhiều biến thể câu hỏi
  2. Encode           — dense + sparse song song
  3. Qdrant Search    — gửi hybrid query (dense + sparse)
  4. Rerank           — CrossEncoder để sắp xếp lại kết quả
  5. Diversity Filter — loại bỏ tài liệu quá giống nhau
"""

import logging
import asyncio
import unicodedata

import torch
import numpy as np
from qdrant_client import models, AsyncQdrantClient
from sentence_transformers import CrossEncoder

from src.core.model_manager import ModelManager, build_query_sparse_vector
from src.core.config import settings

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def remove_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


# ── RAG Retriever ─────────────────────────────────────────────────────────────

class RAGRetriever:
    def __init__(self):
        self.client = AsyncQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            https=False,
        )

        logger.info("Loading Qwen3 Legal Embedding model ...")
        self.embed_model = ModelManager.get_embed_model()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CrossEncoder reranker on %s ...", device)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    def _expand_query(self, query: str) -> list[str]:
        """Tạo các biến thể của câu hỏi để tăng recall."""
        variants = {
            query.strip(),
            query.lower().strip(),
            remove_accents(query).strip(),
        }
        return [q for q in variants if q]

    async def search(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Hybrid search: Dense (Qwen3 4096-dim) + Sparse (BM25 hash).
        Trả về list dict chứa content + metadata cho LLM.
        """
        queries = self._expand_query(query)

        # ── 1. Encode dense vectors (off event-loop thread) ──────────────────
        dense_vecs: np.ndarray = await asyncio.to_thread(
            self.embed_model.encode, queries
        )

        # ── 2. Build sparse vectors ───────────────────────────────────────────
        sparse_vecs = [build_query_sparse_vector(q) for q in queries]

        # ── 3. Qdrant hybrid query (parallel) ────────────────────────────────
        search_tasks = []
        for i, q in enumerate(queries):
            dense_list = dense_vecs[i].tolist()
            sp = sparse_vecs[i]

            task = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    # Dense prefetch
                    models.Prefetch(
                        query=dense_list,
                        using="dense",
                        limit=50,
                    ),
                    # Sparse prefetch (BM25)
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sp["indices"],
                            values=sp["values"],
                        ),
                        using="text_sparse",
                        limit=50,
                    ),
                ],
                # Hybrid fusion via RRF
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=50,
                with_payload=True,
            )
            search_tasks.append(task)

        all_hits: dict = {}
        try:
            results = await asyncio.gather(*search_tasks)
            for response in results:
                for hit in response.points:
                    # Deduplicate by point ID — keep highest score
                    if hit.id not in all_hits or hit.score > all_hits[hit.id].score:
                        all_hits[hit.id] = hit
        except Exception as e:
            logger.error("Error searching in collection '%s': %s", collection_name, e)
            raise RuntimeError("Không thể truy cập dữ liệu của session này.")

        if not all_hits:
            logger.warning("No hits returned from Qdrant.")
            return []

        unique_hits = list(all_hits.values())
        logger.info("Qdrant returned %d unique hits before rerank.", len(unique_hits))

        # ── 4. CrossEncoder Rerank ────────────────────────────────────────────
        passages = [hit.payload.get("content_text", "") for hit in unique_hits]
        rerank_results = await asyncio.to_thread(
            self.reranker.rank,
            query,
            passages,
            return_documents=True,
            top_k=min(len(passages), top_k * 4),
        )

        # ── 5. Semantic Diversity Filter ──────────────────────────────────────
        top_candidate_indices = [
            res["corpus_id"]
            for res in rerank_results
            if res["score"] >= score_threshold
        ]
        top_texts = [
            unique_hits[idx].payload.get("content_text", "")
            for idx in top_candidate_indices
        ]

        if not top_texts:
            return []

        # Re-encode để tính cosine similarity cho diversity filter
        candidate_embs: np.ndarray = await asyncio.to_thread(
            self.embed_model.encode, top_texts
        )

        final_docs = []
        selected_embs = []

        for idx, text in enumerate(top_texts):
            original_hit = unique_hits[top_candidate_indices[idx]]
            current_emb = candidate_embs[idx]

            if not selected_embs:
                final_docs.append(self._format_hit(original_hit, rerank_results[idx]["score"]))
                selected_embs.append(current_emb)
                continue

            # Cosine similarity với các doc đã chọn
            selected_arr = np.array(selected_embs)
            sims = np.dot(selected_arr, current_emb) / (
                np.linalg.norm(selected_arr, axis=1) * np.linalg.norm(current_emb) + 1e-8
            )

            # Thêm vào nếu đủ đa dạng (sim < 0.85)
            if np.max(sims) < 0.85:
                final_docs.append(self._format_hit(original_hit, rerank_results[idx]["score"]))
                selected_embs.append(current_emb)

            if len(final_docs) >= top_k:
                break

        logger.info("Retrieved %d diverse documents after rerank.", len(final_docs))
        return final_docs

    def _format_hit(self, hit, score: float) -> dict:
        """Chuẩn hoá kết quả trả về cho LLM và submission."""
        return {
            "content":          hit.payload.get("content_text", ""),
            "score":            float(score),
            "relevant_doc":     hit.payload.get("relevant_doc", ""),
            "relevant_article": hit.payload.get("relevant_article", ""),
            "law_id":           hit.payload.get("law_id", ""),
            "law_name":         hit.payload.get("law_name", ""),
            "source_article_no": hit.payload.get("source_article_no", ""),
        }
