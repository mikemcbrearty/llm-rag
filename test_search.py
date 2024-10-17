import json

from embeddings_search import EmbeddingsSearch
from full_text_search import FullTextSearch


def test_search(searches, questions, top_k=3, verbose=False):
    query_count = 0
    retrieval_success_count = 0
    items_retrieved = 0
    for q in question_objs:
        if q['section'] == "":
            continue

        retrieval_outcomes = []
        if verbose:
            print(q['section'], q['question'])
        for search in searches:
            result = search.query(q['question'], top_k=top_k)
            sections = []
            for elem in result:
                i = elem.find("—")
                section = elem[:i].strip()
                sections.append(section)
            retrieval_outcome = q['section'] in sections
            retrieval_outcomes.append(retrieval_outcome)
            items_retrieved += len(sections)
            if verbose:
                print(f"{sections=} {retrieval_outcome=}")

        query_count += 1
        retrieval_success = True in retrieval_outcomes
        if retrieval_success:
            retrieval_success_count += 1
    precision = retrieval_success_count / items_retrieved
    recall = retrieval_success_count / query_count
    f1 = 2 * (precision * recall) / (precision + recall)

    containsEmbeddingsSearch = any(type(s) is EmbeddingsSearch for s in searches)
    containsFullTextSearch = any(type(s) is FullTextSearch for s in searches)
    if containsEmbeddingsSearch and containsFullTextSearch:
        search_type = "Combined Search:"
    elif containsEmbeddingsSearch:
        search_type = "EmbeddingsSearch:"
    else:
        search_type = "FullTextSearch:"

    print(f"{search_type:20} (top_k={top_k}) precision: {precision:3.0%}, recall: {recall:3.0%}, f1: {f1:3.0%}")


if __name__ == '__main__':
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

    for s in searches:
        for top_k in top_ks:
            test_search(s, question_objs, top_k=top_k)
