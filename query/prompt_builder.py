def build_rag_prompt(
        question : str,
        text_results : list,
        image_results : list
) -> str:
    
    # prompt are created in such a manner that we number each source so that Ollama can cite them. ["TEXT SOURCE 2"]

    context_parts = []

    for i, results in enumerate(text_results):
        page = results["metadata"].get("page_number","?") # using get so that if value is not present, there is not error. ? is returned. 
        file_name = results["metadata"].get("source_filename","unknown")
        score = results["score"]

        context_parts.append(
            f"[TEXT SOURCE {i+1}]\n"
            f"File : {file_name}, Page : {page}, Relevance : {score:.2f} \n"
            f"{results['text']}"
        )
    for j, results in enumerate(image_results):
        page = results["metadata"].get("page_number","?")
        file_name = results["metadata"].get("source_filename","unknown")
        score = results["score"]

        context_parts.append(
            f"IMAGE SOURCE {len(text_results)+j+1}\n"
            f"File : {file_name}, Page: {page}, Relevance: {score:.2f}\n"
            f"[A relevant image was found on this page.]"
        )

    full_context = "\n\n".join(context_parts)

    prompt = f""" You are a precise document assistant. Answer the user's question using only the document excerpts provided below.
INSTRUCTIONS:
- Use ONLY the information in the DOCUMENT CONTEXT section below to answer.
- Do not use any knowledge from outside these documents.
- If the context does not contain enough information, say: "The provided documents do not contain enough information to answer this question."
- Cite sources in your answer using their labels, e.g. [TEXT SOURCE 1] or [IMAGE SOURCE 3].
- Be clear and concise.

DOCUMENT CONTEXT:
{full_context}

USER QUESTION:
{question}

YOUR ANSWER:
"""
    
    return prompt