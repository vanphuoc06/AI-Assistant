import asyncio
from qdrant_client import AsyncQdrantClient

async def main():
    client = AsyncQdrantClient(host="34.84.237.184", port=6333, api_key="6c635eee6cb87fe9838419cf03b7dc60591355456c4c8326d33c3d2394353705", https=False, timeout=30.0)
    print("Fetching collection info...")
    try:
        info = await client.get_collection("Dataset_Hybrid_BGE_M3_BM25_V1")
        print("Status:", info.status)
        print("Optimizer Status:", info.optimizer_status)
        print("Points count:", getattr(info, "points_count", None))
        print("Indexed vectors:", getattr(info, "indexed_vectors_count", None))
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
