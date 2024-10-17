from datetime import datetime
import json
import logging
import math
import requests

from embeddings_search import EmbeddingsSearch
from full_text_search import FullTextSearch


def test(questions, searches=None, search_top_k=3, verbose=False, logger=None):
    if not searches:
        rag_type = "as is"
    else:
        containsEmbeddingsSearch = any(type(s) is EmbeddingsSearch for s in searches)
        containsFullTextSearch = any(type(s) is FullTextSearch for s in searches)
        if containsEmbeddingsSearch and containsFullTextSearch:
            rag_type = "supplemented with full text search and embeddings search"
        elif containsEmbeddingsSearch:
            rag_type = "supplemented with embeddings search"
        else:
            rag_type = "supplemented with full text search"
    announcement = f"Test LLM {rag_type:40} (top_k={search_top_k}):"
    print(announcement)
    logger.info(announcement)

    accurate_response = 0
    for i, q in enumerate(questions):
        if not searches:
            prompt = q['question']
            print(f"prompt [{i+1}]: {prompt}")
            if logger:
                logger.info(f"prompt [{i+1}]: {prompt}")
        else:
            search_results = []
            for s in searches:
                results = s.query(q['question'], top_k=search_top_k)
                for r in results:
                    if r not in search_results:
                        search_results.append(r)
            intro = "The following context is excerpts from ACI 318-08, the American Concrete Institute Building Code Requirements for Structural Concrete. These excerpts may be relevant for answering a question given below. Use this context to help answer the question if it is relevant.\nContext:\n"

            prompt = f"{intro}{"\n".join(search_results)}\nQuestion: {q['question']}"
            abbrev_prompt = f"{intro}{"\n".join(f"{r[:100]}..." for r in search_results)}\nQuestion: {q['question']}"
            print(abbrev_prompt)
            if logger:
                logger.info(f"prompt [{i+1}]: {prompt}")

        response = llama3_1(prompt)
        if logger:
            logger.info(f"response [{i+1}]: {response}")

        contains_expected = False
        for exp in q['expected']:
            if exp in response.lower():
                contains_expected = True
                accurate_response += 1
                break
        print(f"expected [{i+1}]: {q['expected']}, test case outcome: {contains_expected}")
        if logger:
            logger.info(f"expected [{i+1}]: {q['expected']}")
            logger.info(f"test case outcome [{i+1}]: {contains_expected}\n")

    out = f"Test LLM {rag_type:40} (top_k={search_top_k}): accuracy: {accurate_response/len(questions):3.0%}"
    print(f"The LLM gave a correct response for {accurate_response} of {len(questions)} questions.")
    print(out)
    if logger:
        logger.info(f"The LLM gave a correct response for {accurate_response} of {len(questions)} questions.")
        logger.info(out)

    return out


def llama3_1(prompt:str) -> str:
    """
    Adapted from https://llama.meta.com/docs/llama-everywhere/running-meta-llama-on-mac/
    """
    data = {
        "model": "llama3.1",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    ollama_url = "http://localhost:11434/api/chat"
    response = requests.post(ollama_url, headers=headers, json=data)
    return response.json()["message"]["content"]


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    ts = str(math.floor(datetime.now().timestamp()))
    log_file =  f"logs/test_{ts}.log"
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

    with open("questions/questions.jsonl", 'r') as f:
        question_objs = [json.loads(line) for line in f]

    emb_search = EmbeddingsSearch()
    ft_search = FullTextSearch()

    searches = [
        [ft_search],
        [emb_search],
        [ft_search, emb_search],
    ]
    top_ks = [1, 3, 5, 7]

    results = []
    results.append(test(question_objs, logger=logger))
    for s in searches:
        for top_k in top_ks:
            results.append(test(question_objs, searches=s, search_top_k=top_k, logger=logger))

    for result in results:
        print(result)
        logger.info(result)
