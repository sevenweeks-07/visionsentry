from qdrant_client import QdrantClient
client = QdrantClient(':memory:')
print("Methods in QdrantClient:")
for m in dir(client):
    if not m.startswith('_'):
        print(f" - {m}")
