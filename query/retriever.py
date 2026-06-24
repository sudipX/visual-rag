import numpy as np
from embeddings.text_embedder import embed_text_chunk
from embeddings.clip_embedder import embed_text_clip


def retrieve_relevant_context(
        question: str,
        text_collection,
        image_collection,
        top_k_text: int = 5,
        top_k_images: int = 2
) -> dict:

    text_query_vec = embed_text_chunk(question)
    text_results = []

    if text_collection.count() > 0:
        raw = text_collection.query(
            query_embeddings=[text_query_vec.tolist()],
            n_results=min(text_collection.count(), top_k_text),
            include=["documents", "metadatas", "distances"]
        )
        for i in range(len(raw["documents"][0])):
            text_results.append({
                "text": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],  
                "score": round(1.0 - raw["distances"][0][i], 4)
            })

    image_query_vec = embed_text_clip(question)
    img_results = []

    if image_collection.count() > 0:
        raw = image_collection.query(
            query_embeddings=[image_query_vec.tolist()],
            n_results=min(image_collection.count(), top_k_images),
            include=["documents", "metadatas", "distances"]
        )
        for i in range(len(raw["documents"][0])):
            img_results.append({
                "image_path": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "score": round(1.0 - raw["distances"][0][i], 4)  
            })

    return {
        "text_results":  text_results,
        "image_results": img_results
    }