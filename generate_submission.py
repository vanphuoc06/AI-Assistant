"""
Generate Submission Script
==========================
Đọc file R2AIStage1DATA.json, chạy RAG pipeline cho từng câu hỏi,
và sinh ra file results.json + submission.zip theo đúng format BTC.

Format đầu ra (tuân thủ Thông tin cuộc thi):
[
  {
    "id": <int>,
    "question": "<str>",
    "answer": "<str>",
    "relevant_docs": ["<mã văn bản>|<tên văn bản>"],
    "relevant_articles": ["<mã văn bản>|<tên văn bản>|<điều>"]
  }
]
"""

import os
import sys
import json
import asyncio
import zipfile
import logging
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.search_engine import RAGRetriever
from src.generation.generator import RAGGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
    handlers=[
        logging.FileHandler("generate_submission.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("SubmissionGenerator")


async def generate_answer_non_stream(generator: RAGGenerator, query: str, contexts: list) -> str:
    """Collect all streamed chunks into a single answer string."""
    answer_parts = []
    try:
        async for chunk in generator.generate_stream(query, contexts):
            answer_parts.append(chunk)
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
    return "".join(answer_parts)


def extract_relevant_docs(retrieved_docs: list) -> list:
    """
    Trích xuất danh sách relevant_docs (unique) từ kết quả retrieve.
    Format: "<mã văn bản>|<tên văn bản>"
    Lấy trực tiếp từ trường relevant_doc trong payload Qdrant.
    """
    seen = set()
    result = []
    for doc in retrieved_docs:
        rd = doc.get("relevant_doc", "")
        if rd and rd not in seen:
            seen.add(rd)
            result.append(rd)
    return result


def extract_relevant_articles(retrieved_docs: list) -> list:
    """
    Trích xuất danh sách relevant_articles (unique) từ kết quả retrieve.
    Format: "<mã văn bản>|<tên văn bản>|<điều>"
    Lấy trực tiếp từ trường relevant_article trong payload Qdrant.
    """
    seen = set()
    result = []
    for doc in retrieved_docs:
        ra = doc.get("relevant_article", "")
        if ra and ra not in seen:
            seen.add(ra)
            result.append(ra)
    return result


async def run_pipeline(
    dataset_path: str,
    output_path: str,
    collection_name: str,
    top_k: int = 5,
    checkpoint_path: str = None,
):
    """Main pipeline: Retrieve -> Generate -> Format -> Save."""

    # Load dataset
    logger.info(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Total questions: {len(dataset)}")

    # Initialize models
    logger.info("Initializing RAG Retriever...")
    retriever = RAGRetriever()
    logger.info("Initializing RAG Generator...")
    generator = RAGGenerator()

    # Load checkpoint if exists
    if checkpoint_path is None:
        checkpoint_path = output_path.replace(".json", "_checkpoint.jsonl")

    results = []
    processed_ids = set()
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    results.append(item)
                    processed_ids.add(item["id"])
        logger.info(f"Resumed {len(results)} items from checkpoint.")

    # Process each question
    with open(checkpoint_path, "a", encoding="utf-8") as ckpt_file:
        remaining = [item for item in dataset if item["id"] not in processed_ids]
        logger.info(f"Remaining questions to process: {len(remaining)}")

        for item in tqdm(remaining, desc="Generating submissions"):
            q_id = item["id"]
            question = item["question"]

            try:
                # Step 1: Retrieve relevant documents
                retrieved_docs = await retriever.search(
                    query=question,
                    collection_name=collection_name,
                    top_k=top_k,
                )

                # Step 2: Generate answer using LLM
                if retrieved_docs:
                    answer = await generate_answer_non_stream(generator, question, retrieved_docs)
                else:
                    answer = "Dựa trên tài liệu hiện tại, tôi không tìm thấy thông tin để trả lời câu hỏi này."

                # Step 3: Extract relevant_docs and relevant_articles from payload
                relevant_docs = extract_relevant_docs(retrieved_docs)
                relevant_articles = extract_relevant_articles(retrieved_docs)

                result_item = {
                    "id": q_id,
                    "question": question,
                    "answer": answer,
                    "relevant_docs": relevant_docs,
                    "relevant_articles": relevant_articles,
                }

            except Exception as e:
                logger.error(f"Error processing question id={q_id}: {e}")
                result_item = {
                    "id": q_id,
                    "question": question,
                    "answer": "",
                    "relevant_docs": [],
                    "relevant_articles": [],
                }

            results.append(result_item)

            # Save checkpoint
            ckpt_file.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            ckpt_file.flush()

    # Sort results by id to match input order
    results.sort(key=lambda x: x["id"])

    # Save final results.json
    logger.info(f"Saving results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Create submission.zip
    zip_path = output_path.replace(".json", ".zip").replace("results", "submission")
    if zip_path == output_path:
        zip_path = os.path.join(os.path.dirname(output_path), "submission.zip")

    logger.info(f"Creating submission zip: {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_path, "results.json")  # Flatten: results.json at root

    logger.info(f"Done! Submission zip created at: {zip_path}")
    logger.info(f"Total items: {len(results)}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("Checkpoint file cleaned up.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate competition submission file.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/R2AIStage1DATA.json",
        help="Path to R2AIStage1DATA.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output path for results.json",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="Dataset_Hybrid_BGE_M3_BM25_V1",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top documents to retrieve per question",
    )

    args = parser.parse_args()

    asyncio.run(
        run_pipeline(
            dataset_path=args.dataset,
            output_path=args.output,
            collection_name=args.collection,
            top_k=args.top_k,
        )
    )
