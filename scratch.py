import asyncio
from src.retrieval.search_engine import RAGRetriever

async def main():
    try:
        retriever = RAGRetriever()
        docs = await retriever.search(query="Hỏi về hội đồng quản trị", collection_name="Dataset_Hybrid_BGE_M3_BM25_V1", top_k=5)
        print("Success:", docs)
    except Exception as e:
        print("ERROR:", repr(e))

if __name__ == "__main__":
    asyncio.run(main())
