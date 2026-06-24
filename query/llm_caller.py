from ollama import Client
from query.prompt_builder import build_rag_prompt

client = Client()
MODEL = "llama3.2:3b"


def call_ollama(prompt: str) -> str:
    response = client.generate(model=MODEL, prompt=prompt)
    return response["response"]


def answer_question(
    question: str,
    text_results: list,
    image_results: list
) -> dict:
    prompt = build_rag_prompt(question, text_results, image_results)
    answer = call_ollama(prompt)

    sources = []
    for i, result in enumerate(text_results):
        sources.append({
            "type": "text",
            "source_number": i + 1,
            "page_number": result["metadata"].get("page_number"),
            "source_filename": result["metadata"].get("source_filename"),
            "relevance_score": result["score"],
            "excerpt": result["text"][:200] + "..."
        })

    for j, img in enumerate(image_results):
        sources.append({
            "type": "image",
            "source_number": len(text_results) + j + 1,
            "page_number": img["metadata"].get("page_number"),
            "source_filename": img["metadata"].get("source_filename"),
            "relevance_score": img["score"],
            "image_path": img.get("image_path")
        })

    return {
        "answer": answer,
        "sources": sources,
        "question": question
    }