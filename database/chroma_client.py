import chromadb

DB_PATH = "./chromadb"
client = chromadb.PersistentClient(path=DB_PATH) # persistant is used so that the data is stored permanently in the given location.

text_collection = client.get_or_create_collection(
    name = "text_chunks",
    metadata={"hnsw":"cosine"}
)

image_collection = client.get_or_create_collection(
    name = "image_chunks",
    metadata={"hnsw":"cosine"}
)

print(f"text_chunks collection : {text_collection.count()} items")
print(f"image_chunks collection : {image_collection.count()} items")