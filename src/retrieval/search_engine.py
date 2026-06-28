import logging
import torch
import unicodedata
import asyncio
import numpy as np
from qdrant_client import models, AsyncQdrantClient
from sentence_transformers import CrossEncoder

from src.utils.nlp_utils import segment_vietnamese
from src.core.model_manager import ModelManager
from src.core.config import settings

logger = logging.getLogger(__name__)


# remove text accents
def remove_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


# main rag retriever class
class RAGRetriever:
    def __init__(self):
        self.client = AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, api_key=settings.QDRANT_API_KEY, https=False, timeout=300.0)
        logger.info("Loading BGE-M3 (Embedding)...")
        self.embed_model = ModelManager.get_embed_model()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Reranker on device: {device}...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=1024, torch_dtype=torch.float16)

    def _expand_query(self, query: str):
        queries = {query.strip(), query.lower().strip(), remove_accents(query).strip()}
        return list(q for q in queries if q)  # return list to keep index order

    def _normalize_sparse(self, sparse_dict):
        if not sparse_dict:
            return {"indices": [], "values": []}
        max_val = max(sparse_dict.values()) or 1.0
        indices = []
        values = []
        # sort to ensure deterministic ordering (not strictly required by Qdrant but good practice)
        for k, v in sorted(sparse_dict.items(), key=lambda x: int(x[0])):
            indices.append(int(k))
            values.append(float(v / max_val))
        return {"indices": indices, "values": values}

    # async hybrid search
    async def search(
        self, query: str, collection_name: str, top_k: int = 5, score_threshold: float = 0.0
    ):
        segmented_query = segment_vietnamese(query)
        queries = self._expand_query(segmented_query)
        all_hits = {}

        # batch encode simultaneously
        emb = await asyncio.to_thread(
            self.embed_model.encode, queries, return_dense=True, return_sparse=True, batch_size=1, max_length=1024
        )

        # async search parallel qdrant requests with correct Hybrid RRF approach
        search_tasks = []
        for i, q in enumerate(queries):
            dense_query = emb["dense_vecs"][i].tolist()
            sparse_query = self._normalize_sparse(emb["lexical_weights"][i])

            task = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense_query,
                        using="dense",
                        limit=max(20, top_k * 2),
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query["indices"],
                            values=sparse_query["values"],
                        ),
                        using="bm25",
                        limit=max(20, top_k * 2),
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=max(20, top_k * 2),
                with_payload=True,
            )
            search_tasks.append(task)

        try:
            # receive results in parallel
            results = await asyncio.gather(*search_tasks)
            for response in results:
                for hit in response.points:
                    if hit.id not in all_hits or hit.score > all_hits[hit.id].score:
                        all_hits[hit.id] = hit
        except Exception as e:
            logger.error(f"Error searching in collection '{collection_name}': {e}")
            raise RuntimeError("Không thể truy cập dữ liệu của session này.")

        if not all_hits:
            return []

        unique_hits = list(all_hits.values())

        # rerank
        passages = [
            hit.payload.get("content_text", "")
            for hit in unique_hits
        ]
        rerank_results = await asyncio.to_thread(
            self.reranker.rank,
            query,
            passages,
            return_documents=True,
            top_k=min(len(passages), top_k * 4),  # get more for filter buffer
            batch_size=1,
        )

        # semantic diversity filter
        final_docs = []
        selected_embs = []

        # get dense vectors to calculate similarity
        top_candidates_idx = [
            res["corpus_id"] for res in rerank_results if res["score"] >= score_threshold
        ]
        top_texts = [
            unique_hits[idx].payload.get("content_text", "") for idx in top_candidates_idx
        ]

        if not top_texts:
            return []

        candidate_embs = await asyncio.to_thread(
            self.embed_model.encode, top_texts, return_dense=True, batch_size=1, max_length=1024
        )
        dense_embs = candidate_embs["dense_vecs"]

        for idx, text in enumerate(top_texts):
            original_hit = unique_hits[top_candidates_idx[idx]]
            current_emb = dense_embs[idx]

            # add doc if no docs selected yet
            if not selected_embs:
                final_docs.append(self._format_hit(original_hit, rerank_results[idx]["score"]))
                selected_embs.append(current_emb)
                continue

            # calc cosine sim with selected docs
            sims = np.dot(selected_embs, current_emb) / (
                np.linalg.norm(selected_embs, axis=1) * np.linalg.norm(current_emb)
            )

            # select if sim is low
            if np.max(sims) < 0.85:
                final_docs.append(self._format_hit(original_hit, rerank_results[idx]["score"]))
                selected_embs.append(current_emb)

            if len(final_docs) >= top_k:
                break

        logger.info(f"Retrieved {len(final_docs)} diverse documents.")
        return final_docs

    # format search result
    def _format_hit(self, hit, score):
        return {
            "content": hit.payload.get("content_text", ""),
            "score": float(score),
            "relevant_doc": hit.payload.get("relevant_doc", ""),
            "relevant_article": hit.payload.get("relevant_article", ""),
            "law_id": hit.payload.get("law_id", ""),
            "law_name": hit.payload.get("law_name", ""),
            "source_article_no": hit.payload.get("source_article_no", ""),
        }
