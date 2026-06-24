import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

text_model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded. Embedding dimension : {text_model.get_embedding_dimension()}")

def embed_text_chunk(text : str) -> np.ndarray:
    # Convert a single text string into a 384-dimensional embedding vector
    # Suitable for one sentence to few paragraphs

    embedding = text_model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embedding

def embed_text_batch(text:list) -> np.ndarray:
    # Embed a list of texts in one efficient batch operations
    # like when we have a whole PDF (a lot of chunks)

    embedding = text_model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32
    )

    return embedding  # shape (N,384) where N = len(text)


if __name__=="__main__":

    #Single Embedding
    text = "The mitochondria is the powerhouse of the cell."
    vec = embed_text_chunk(text)
    print(vec.shape)
    print(np.linalg.norm(vec))

    # Batch Embedding
    texts = ["Machine Learning is the subset of Artificial Intelligence.",
             "Deep Learning use neural networks with many layers",
             "The recipe requires two cups of flour and one egg"]
    
    vecs = embed_text_batch(texts)
    print(vecs.shape)

    print(np.dot(vecs[0],vecs[1]))
    print(np.dot(vecs[0],vecs[2]))
