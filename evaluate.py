"""Score the search and the refusal behaviour on a public benchmark (SQuAD 2.0 dev).

  python evaluate.py
Steps: rank chunks for 3,000 questions with 4 methods, tune the refusal threshold on half the articles (dev),
report everything on the other half (test). Writes outputs/results.json; report.py turns that into a page.
"""
import json
import os
import pickle
import time
import numpy as np

from docqa import corpus
from docqa.search import Index, ROOT
from docqa.qa import QA, split_sentences

K = 10
OUT = os.path.join(ROOT, "outputs")
CACHE = os.path.join(ROOT, "data", "cache", "eval_rankings_1000.pkl")


def rank_all(index: Index, questions):
    """For every question, the ranked chunk ids from each method (admin view: no permission filter)."""
    mask = index.allowed_mask(None)
    qvecs = index.embed_queries([q.text for q in questions])
    out = []
    t = {"bm25": 0.0, "dense": 0.0, "hybrid": 0.0, "hybrid+rerank": 0.0}
    for n, (q, qv) in enumerate(zip(questions, qvecs)):
        t0 = time.perf_counter(); b, _ = index.bm25_search(q.text, mask, K); t["bm25"] += time.perf_counter() - t0
        t0 = time.perf_counter(); d, _ = index.dense_search(qv, mask, K); t["dense"] += time.perf_counter() - t0
        t0 = time.perf_counter(); h, _ = index.hybrid_search(q.text, qv, mask, 20); t["hybrid"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        r, rs = index.rerank(q.text, h[:8], K) if index.reranker is not None else (h[:K], np.array([0.0]))
        t["hybrid+rerank"] += time.perf_counter() - t0 + (t["hybrid"] - 0) * 0
        out.append(dict(qid=q.qid, bm25=list(map(int, b)), dense=list(map(int, d)), hybrid=list(map(int, h[:K])),
                        rerank=list(map(int, r)), top_score=float(rs[0]) if len(rs) else float("-inf")))
        if n % 500 == 0:
            print(f"  ranked {n}/{len(questions)}")
    ms = {k: round(v / len(questions) * 1000, 2) for k, v in t.items()}
    return out, ms


def retrieval_metrics(questions, ranks, method):
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rr, n = 0.0, 0
    for q, r in zip(questions, ranks):
        if not q.answerable:
            continue
        n += 1
        order = r[method]
        for k in hits:
            hits[k] += q.chunk_id in order[:k]
        if q.chunk_id in order:
            rr += 1.0 / (order.index(q.chunk_id) + 1)
    return {f"recall@{k}": round(v / n, 4) for k, v in hits.items()} | {"mrr@10": round(rr / n, 4), "questions": n}


def auroc(pos, neg):
    """Probability that a random answerable question gets a higher confidence than a random unanswerable one."""
    pos, neg = np.array(pos), np.array(neg)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_threshold(questions, ranks):
    """Pick the confidence cut-off that maximises balanced accuracy on the dev half."""
    conf = np.array([r["top_score"] for r in ranks])
    ans = np.array([q.answerable for q in questions])
    best = (-1, None)
    for t in np.unique(np.quantile(conf, np.linspace(0.02, 0.98, 97))):
        tpr = (conf[ans] >= t).mean()            # answerable questions we answer
        tnr = (conf[~ans] < t).mean()            # unanswerable questions we refuse
        bal = (tpr + tnr) / 2
        if bal > best[0]:
            best = (bal, float(t))
    return best[1]


def abstention_metrics(questions, ranks, threshold):
    conf = np.array([r["top_score"] for r in ranks])
    ans = np.array([q.answerable for q in questions])
    top1_ok = np.array([q.chunk_id == r["rerank"][0] if r["rerank"] else False for q, r in zip(questions, ranks)])
    answers, refuses = conf >= threshold, conf < threshold
    return dict(
        threshold=round(threshold, 3),
        answer_rate_on_answerable=round(float(answers[ans].mean()), 4),
        refusal_rate_on_unanswerable=round(float(refuses[~ans].mean()), 4),
        balanced_accuracy=round(float((answers[ans].mean() + refuses[~ans].mean()) / 2), 4),
        confidence_auroc=round(auroc(conf[ans], conf[~ans]), 4),
        end_to_end_correct=round(float(((ans & answers & top1_ok) | (~ans & refuses)).mean()), 4),
    )


def sentence_hit_rate(index: Index, qa: QA, questions, ranks):
    """For answerable questions whose top chunk is the right one: does the picked sentence contain a gold answer?"""
    hit = n = 0
    for q, r in zip(questions, ranks):
        if not q.answerable or not r["rerank"] or r["rerank"][0] != q.chunk_id:
            continue
        n += 1
        s = qa._best_sentence(q.text, q.chunk_id).lower()
        hit += any(a.lower() in s for a in q.answers)
    return dict(questions=n, sentence_contains_gold_answer=round(hit / max(n, 1), 4))


def main():
    os.makedirs(OUT, exist_ok=True)
    chunks, allq = corpus.load()
    qs = corpus.sample_questions(allq)
    print(f"{len(chunks)} chunks, {len(qs)} questions (balanced answerable / unanswerable)")
    index = Index(chunks)
    if os.path.exists(CACHE):
        ranks, ms = pickle.load(open(CACHE, "rb"))
    else:
        ranks, ms = rank_all(index, qs)
        pickle.dump((ranks, ms), open(CACHE, "wb"))
    by_id = {q.qid: i for i, q in enumerate(qs)}
    dev, test = corpus.split_by_article(qs)
    rk = lambda qq: [ranks[by_id[q.qid]] for q in qq]

    results = dict(corpus=dict(chunks=len(chunks), questions=len(qs), dev_questions=len(dev), test_questions=len(test)),
                   latency_ms_per_query=ms, retrieval={}, notes=[])
    for name, key in [("keyword only (BM25)", "bm25"), ("embeddings only", "dense"),
                      ("hybrid (BM25 + embeddings, RRF)", "hybrid"), ("hybrid + reranker", "rerank")]:
        results["retrieval"][name] = retrieval_metrics(test, rk(test), key)

    thr = best_threshold(dev, rk(dev))
    results["abstention_test"] = abstention_metrics(test, rk(test), thr)
    qa = QA(index, thr)
    results["answer_sentence"] = sentence_hit_rate(index, qa, test, rk(test))
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
