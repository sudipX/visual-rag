def chunk_text(text:str, chunk_size:int=400,chunk_overlap:int=80)->list:
    # split the text string into overlapping chunks of approximately chunk_size words.

    words = text.split()

    if len(words)<=chunk_size:
        return [text]
    
    chunks =[]

    step = chunk_size-chunk_overlap

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append(" ".join(chunk_words))

        start+=step

        remaining = len(words)-start
        
        if 0 < remaining < chunk_size //4:
            chunks.append(" ".join(words[start:]))
            break
    
    return chunks

def chunk_page_text(
        page_text : str,
        page_number : int,
        source_filename : str,
        chunk_size : int = 400,
        chunk_overlap : int =80
) -> list:
    raw_chunks = chunk_text(page_text, chunk_size, chunk_overlap)

    safe_name = source_filename.replace(" ","_").replace(".","_")

    result = []

    for idx, chunk in enumerate(raw_chunks):
        chunk_id = f"{safe_name}_page{page_number}_chunk{idx}"
        result.append({
            "chunk_id" : chunk_id,
            "text" : chunk,
            "page_number" : page_number,
            "source_filename" : source_filename,
            "chunk_index" : idx
        })
    
    return result

# if __name__ == "__main__":
#     sample = " ".join([f"word{i}" for i in range(1000)])
#     chunks = chunk_text(sample, chunk_size=400, chunk_overlap=80)

#     print(f"Input : 1000 words")
#     print(f"Output : {len(chunks)} chunks")
#     for i,c in enumerate(chunks):
#         print(f"Chunk {i} : {len(c.split())} words.")