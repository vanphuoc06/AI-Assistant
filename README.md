# 🇻🇳 Vietnamese RAG System (Production-Ready)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid_Search-red.svg)
![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5-black.svg)
![Redis](https://img.shields.io/badge/Redis-Caching-dc382d.svg)

> A production-oriented Retrieval-Augmented Generation (RAG) system designed with explicit trade-offs in retrieval quality, latency, and evaluation.

---

## Demo
<p align="center">
  <img src="https://raw.githubusercontent.com/NamSyntax/vietnamese-rag-system/master/docs/Demo.gif" width="95%"/>
</p>

---
## Features

- Hybrid retrieval (dense + sparse) with Reciprocal Rank Fusion (RRF)
- Query expansion for Vietnamese normalization (case, accent removal)
- Cross-encoder reranking (BGE-Reranker) for precision optimization
- Semantic diversity filtering to remove redundant context
- Async FastAPI pipeline with streaming responses
- Redis-based exact-match caching for latency reduction
- LLM-as-a-judge evaluation with automated root cause analysis

## System Architecture & Workflow

The system is built with **FastAPI** (backend) and **Streamlit** (frontend), integrating the following core workflows:

1. **Ingestion (`src/ingestion`)**: Parses PDFs using `PyMuPDF`. Instead of standard token splitting, it applies structural chunking with length constraints and overlapping. Vietnamese text is segmented using `underthesea` for better embedding accuracy.
2. **Vector Store (`src/ingestion/vector_store.py`)**: Asynchronously manages document embeddings in **Qdrant**. It computes both dense vectors and sparse lexical weights using the `BGE-M3` model.
3. **Retrieval (`src/retrieval`)**: 
   - **Query Expansion:** Normalizes and expands the user query.
   - **Hybrid Search:** Executes concurrent searches on Qdrant using Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
   - **Reranking & Filtering:** Uses `bge-reranker-v2-m3` to score top candidates, followed by a custom Cosine-Similarity Diversity Filter to remove semantically redundant chunks.
4. **Generation (`src/generation`)**: Constructs context-aware prompts with length protection. Streams responses via **Ollama** running `Qwen2.5:7b-instruct`.
5. **Caching (`src/core/cache.py`)**: Uses **Redis** to cache LLM responses and manage async background task statuses (e.g., file upload progress).

<p align="center">
  <img src="https://raw.githubusercontent.com/NamSyntax/vietnamese-rag-system/master/docs/RAGSystemArchitecture.png" width="95%"/>
</p>

## Key Design Insights

This system is not just a RAG implementation — it is built around several critical observations:

- **Hybrid retrieval alone is insufficient**  
  → Dense + sparse improves recall, but introduces semantic noise.

- **Reranking is mandatory, not optional**  
  → Cross-encoder reranking significantly improves precision, but adds latency (O(N)).

- **Redundancy is a hidden bottleneck**  
  → Retrieved chunks often contain overlapping information → solved via cosine-similarity diversity filtering.

- **Exact-match caching works surprisingly well**  
  → For document QA, repeated queries are common → simple hashing provides high ROI.

- **Evaluation must be decomposed**  
  → Retrieval and generation are evaluated separately to identify real failure modes.

## Tech Stack

- **Backend:** FastAPI, Python `asyncio`
- **Frontend:** Streamlit
- **Vector Database:** Qdrant (Async Client)
- **Caching:** Redis
- **Models:** BAAI/bge-m3 (Embedding), BAAI/bge-reranker-v2-m3 (Reranking), Qwen2.5:7b-instruct (Generation via Ollama)
- **NLP:** Underthesea (Vietnamese word segmentation)

## What This System Gets Right

- Separates retrieval quality from generation quality  
- Uses reranking to correct hybrid retrieval noise  
- Handles Vietnamese-specific preprocessing (segmentation + normalization)  
- Implements async streaming for better UX  
- Includes a full evaluation pipeline with failure diagnosis (not just metrics)

## Installation

### Prerequisites
- [uv](https://docs.astral.sh/uv/) (Highly Recommend)
- Docker (for Qdrant and Redis)
- [Ollama](https://ollama.com) installed locally.
- Python 3.12+ (managed by `uv`)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NamSyntax/vietnamese-rag-system.git
   cd vietnamese-rag-system
   ```

2. **Install dependencies:**
   With `uv`, all dependencies and the virtual environment are managed automatically:
   ```bash
   uv sync
   ```

3. **Start required services:**
   Ensure Docker and Ollama are running, then pull the necessary models:
   ```bash
   # Run Qdrant and Redis via Docker
   docker run -d -p 6333:6333 qdrant/qdrant
   docker run -d -p 6379:6379 redis
   
   # Pull the local LLM
   ollama pull qwen2.5:7b-instruct
   ```

4. **Environment Configuration:**
   Create a `.env` file in the root directory (refer to the template below):
   ```env
   QDRANT_HOST="localhost"
   QDRANT_PORT=6333
   REDIS_URL="redis://localhost:6379"
   OLLAMA_BASE_URL="http://localhost:11434/api/chat"
   LLM_MODEL_NAME="qwen2.5:7b-instruct"
   GEMINI_API_KEY="your_api_key_for_evaluation"
   ```

## Usage

Open http://localhost:8501 to upload a PDF and start querying.

## 🏆 Cuộc thi AI-Guru: Hướng dẫn nộp bài (Submission Pipeline)

Hệ thống đã được thiết kế lại để tuân thủ 100% quy chế cuộc thi AI-Guru (không sử dụng API đóng như Gemini/OpenAI, chỉ dùng model Local + Open Source dưới 14B tham số).

Để tạo file `results.json` và `submission.zip` nộp cho Ban Tổ Chức, bạn sử dụng script `generate_submission.py`.

### Chạy thử nghiệm nhanh (Test 1-2 câu)
Mình đã tạo sẵn file `test_2_cau.json` chứa 2 câu hỏi mẫu để bạn chạy thử toàn bộ luồng mà không phải chờ đợi lâu:
```bash
uv run generate_submission.py --dataset test_2_cau.json --output test_results.json
```
Lệnh trên sẽ in log quá trình Retrieve & Generate, sau đó tạo ra file `test_results.json` và nén thành `submission.zip` để bạn kiểm tra định dạng.

### Chạy thật toàn bộ dữ liệu cuộc thi
Để chạy thật trên bộ câu hỏi đầy đủ của BTC (2001 câu):
```bash
uv run generate_submission.py --dataset "..\R2AIStage1DATA.json" --output results.json --collection "Dataset_Hybrid_Embbeding_Model_Trained_V1"
```
*Lưu ý: Quá trình này sẽ tốn thời gian. Script có hỗ trợ cơ chế Checkpoint: nếu máy tính mất điện hoặc bạn ấn Ctrl+C để dừng, lần sau chạy lại lệnh trên, script sẽ tự động tiếp tục ở câu hỏi bị ngắt quãng mà không phải chạy lại từ đầu.*




## Evaluation & Benchmarks

To ensure the system's reliability in practical applications, we implemented a comprehensive **LLM-as-a-Judge** evaluation pipeline (`src/evaluation/evaluator.py`). 

### Dataset
The evaluation is conducted on a custom **420-question Vietnamese Legal QA Dataset**, featuring diverse query types including unanswerable questions to test hallucination resistance.
- A 10-row sample is included in `data/sample_evaluation.jsonl` for quick inspection.
- The full, complete dataset is hosted on Hugging Face: [**🤗 NamSyntax/Vietnamese-Legal-QA-RAG**](https://huggingface.co/datasets/NamSyntax/Vietnamese-Legal-QA-RAG)

### Evaluation Methodology
The pipeline independently assesses both the **Retriever** and the **Generator** to pinpoint exact bottlenecks, rather than just grading the final answer.

1. **Semantic Retrieval Evaluation:** Instead of brittle exact-string matching, we utilize a Cross-Encoder (`bge-reranker-v2-m3`) to compute the semantic similarity between the retrieved chunks and the Ground Truth context.
   - **Metrics:** `Hit@K`, `MRR` (Mean Reciprocal Rank), `Recall@K`.

2. **Grounded Generation Evaluation (LLM Judge):** We employ `gemini-2.5-flash` with a strict grading prompt (Score 0-10) to act as an impartial judge.
   - **Faithfulness:** Measures if the generated answer strictly adheres to the retrieved context (heavily penalizing hallucination).
   - **Answer Correctness:** Measures if the generated answer semantically resolves the Ground Truth.
   - **Abstention Accuracy:** Evaluates the system's ability to correctly state "I don't know" when faced with out-of-context queries.

3. **Automated Root Cause Analysis (RCA):**
   The pipeline goes beyond raw numbers by automatically categorizing failure modes. If a query fails, the system diagnoses the root cause:
   - *Retrieval Failure:* The necessary context was completely missing in the Top-K results.
   - *Context Failure:* The context was retrieved but was too truncated or noisy.
   - *Generation Failure:* The context was perfect, but the LLM failed to extract the answer or hallucinated.

## 📊 Evaluation Results

<p align="center">
  <img src="https://raw.githubusercontent.com/NamSyntax/vietnamese-rag-system/master/docs/evaluation_plots/metrics_by_question_type.png" width="100%"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/NamSyntax/vietnamese-rag-system/master/docs/evaluation_plots/error_distribution_bar.png" width="100%"/>
</p>

<p align="left">
  <img src="https://raw.githubusercontent.com/NamSyntax/vietnamese-rag-system/master/docs/evaluation_plots/overall_radar_chart.png" width="60%"/>
</p>

## Known Limitations

- Cross-encoder reranking introduces latency (O(N) over candidates)
- No semantic caching (only exact-match caching)
- BackgroundTasks are not fault-tolerant (no queue / retry mechanism)
- Performance not benchmarked under high concurrency
- Context window limits may truncate long documents

> These trade-offs were intentionally accepted to prioritize system clarity and local deployment simplicity.

## Author

**Vu Hoang Nam (NamSyntax)**  
Email: [namsyntax@gmail.com](mailto:namsyntax@gmail.com)