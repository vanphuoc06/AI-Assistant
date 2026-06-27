import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

async def main():
    client = AsyncQdrantClient(host="34.180.121.34", port=6333, api_key="6c635eee6cb87fe9838419cf03b7dc60591355456c4c8326d33c3d2394353705", https=False, timeout=5.0)
    
    dense_query = [0.1] * 1024
    sparse_indices = [1, 2, 3]
    sparse_values = [0.1, 0.2, 0.3]
    
    try:
        task = await client.query_points(
            collection_name="Dataset_Hybrid_BGE_M3_BM25_V1",
            query=models.SparseVector(
                indices=sparse_indices,
                values=sparse_values,
            ),
            using="bm25",
            limit=5,
            with_payload=True,
        )
        print("Success:", len(task.points))
    except Exception as e:
        print("ERROR TYPE:", type(e))
        print("ERROR REPR:", repr(e))
        print("ERROR STR:", str(e))

if __name__ == "__main__":
    asyncio.run(main())
