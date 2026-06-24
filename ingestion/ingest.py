from pathlib import Path
from ingestion.pdf_extractor import extract_image_by_page, extract_text_by_page
from ingestion.chunker import chunk_page_text
from embeddings.text_embedder import embed_text_batch
from embeddings.clip_embedder import embed_image

IMAGE_CAHCE_DIR = "./extracted_images"

def ingest_pdf(pdf_path:str,text_collection, image_collection) ->dict:

    source_filename = Path(pdf_path).name
    print(f"Ingesting {source_filename}")

    print(f"Step 1/4 : Extracting text")
    pages = extract_text_by_page(pdf_path)

    print(f"Step 2/4 : Chunking text")
    all_chunks = []
    for page in pages:
        page_chunk = chunk_page_text(
            page_text= page["text"],
            page_number=page["page_number"],
            source_filename=source_filename
        )
        all_chunks.extend(page_chunk)
    print(f"Total chunks : {len(all_chunks)}")

    print(f"Step 3/4 : Embed all text chunks in one efficient batch call")

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_text_batch(texts)

    print(f"Step 4a/4 : Storing text chunks in ChromaDB")
    
    if all_chunks:
        text_collection.upsert(
            ids = [c["chunk_id"] for c in all_chunks],
            embeddings = embeddings.tolist(),
            documents = texts,
            metadatas = [
                {
                    "page_number" : c["page_number"],
                    "source_filename" : source_filename,
                    "chunk_index" : c["chunk_index"]
                }
                for c in all_chunks
            ]
        )
    
    print(f"Step 4b/4 : Processing images")

    image_records = extract_image_by_page(pdf_path, IMAGE_CAHCE_DIR)

    img_ids = []
    img_vecs = []
    img_metas = []
    img_docs = []

    for rec in image_records:
        try:
            vec = embed_image(rec["image_path"])
            img_id = (f"{Path(pdf_path).stem}_page{rec["page_number"]}_img{rec["image_index"]}")
            img_ids.append(img_id)
            img_vecs.append(vec.tolist())
            img_metas.append({
                "page_number" : rec["page_number"],
                "source_filename" : source_filename,
                "image_path" : rec["image_path"] 
            })
            img_docs.append(rec["image_path"])
        except Exception as e:
            print(f"Skipped image embedding : {e}")
            continue

    if img_ids:
        image_collection.upsert(
            ids = img_ids,
            embeddings = img_vecs,
            documents = img_docs,
            metadatas = img_metas
        )

    summary = {
        "source_filename" : source_filename,
        "pages_processed" : len(pages),
        "text_chunks_stored" : len(all_chunks),
        "images_stored" : len(img_ids)
    }

    print(f"Done : {summary}")
    return summary 