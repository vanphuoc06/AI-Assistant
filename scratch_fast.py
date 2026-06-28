import asyncio
import time
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

async def main():
    client = AsyncQdrantClient(host="34.84.237.184", port=6333, api_key="6c635eee6cb87fe9838419cf03b7dc60591355456c4c8326d33c3d2394353705", https=False, timeout=60.0)
    
    dense_query = [0.1] * 1024
    sparse_indices = [1, 2, 3]
    sparse_values = [0.1, 0.2, 0.3]
    
    collection_name = "Dataset_Hybrid_BGE_M3_BM25_V1"
    
    print("Testing Dense Only...")
    t0 = time.time()
    try:
        res = await client.query_points(
            collection_name=collection_name,
            query=dense_query,
            using="dense",
            limit=5
        )
        print(f"Dense time: {time.time() - t0:.2f}s, hits: {len(res.points)}")
    except Exception as e:
        print("Dense Error:", repr(e))

    print("Testing Sparse Only...")
    t0 = time.time()
    try:
        res = await client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(indices=sparse_indices, values=sparse_values),
            using="bm25",
            limit=5
        )
        print(f"Sparse time: {time.time() - t0:.2f}s, hits: {len(res.points)}")
    except Exception as e:
        print("Sparse Error:", repr(e))

if __name__ == "__main__":
    asyncio.run(main())
