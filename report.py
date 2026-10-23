"""Build outputs/report.html from outputs/results.json (shared look in theme.py)."""
import json

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

import theme as T

r = json.load(open("outputs/results.json"))
ret = r["retrieval"]
names = list(ret)
a, s, lat = r["abstention_test"], r["answer_sentence"], r["latency_ms_per_query"]
pct = lambda v: f"{v * 100:.1f}%"
best = names[-1]

fig = go.Figure()
metrics = [("recall@1", "Right one first"), ("recall@5", "In top 5"), ("recall@10", "In top 10"), ("mrr@10", "MRR at 10")]
cols = ["#b9bcc8", "#e0803a", "#3d8fb8", "#4b3fd6"]
for name, col in zip(names, cols):
    fig.add_bar(name=name, x=[m[1] for m in metrics], y=[ret[name][m[0]] for m in metrics], marker_color=col,
                text=[f"{ret[name][m[0]]:.2f}" for m in metrics], textposition="outside")
chart = T.fig_html(fig, barmode="group", height=420, yaxis=dict(range=[0, 1.12], title="score", gridcolor="#e9e5da"),
                   legend=dict(orientation="h", y=-0.18), margin=dict(l=60, r=20, t=24, b=80))

ret_rows = [[T.esc(n)] + [f"{ret[n][k]:.3f}" for k in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr@10")] for n in names]
ret_table = T.table(["Method", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR@10"], ret_rows, numeric_cols=(1, 2, 3, 4, 5))

abst_table = T.table(["Measure", "Result"], [
    ["Confidence threshold (chosen on dev articles)", f"{a['threshold']}"],
    ["Answerable questions that got an answer", pct(a["answer_rate_on_answerable"])],
    ["Unanswerable questions correctly refused", pct(a["refusal_rate_on_unanswerable"])],
    ["Balanced accuracy", pct(a["balanced_accuracy"])],
    ["Fully correct (answered with the right paragraph on top, or correctly refused)", pct(a["end_to_end_correct"])],
    ["How well the confidence score separates the two groups (AUROC, 0.5 = coin flip)", f"{a['confidence_auroc']:.2f}"],
], numeric_cols=(1,))

lat_rows = [[("reranker stage (adds to hybrid)" if k == "hybrid+rerank" else k), f"{v} ms"] for k, v in lat.items()]

body = (T.kpis([
    ("Right paragraph in top 5", f"{ret[best]['recall@5']:.0%}", "best method: hybrid + reranker"),
    ("Right paragraph ranked first", f"{ret[best]['recall@1']:.0%}", "best method"),
    ("Answer sentence correct", f"{s['sentence_contains_gold_answer']:.0%}", "when the paragraph is right"),
    ("Unanswerable questions refused", f"{a['refusal_rate_on_unanswerable']:.0%}", "a weakness, see below"),
    ("Confidence AUROC", f"{a['confidence_auroc']:.2f}", "0.5 means a coin flip"),
])
    + T.section("01 / Finding the right paragraph", "Combining methods wins",
                T.card(chart, caption="Answerable test questions only. Higher is better.") + T.card(ret_table),
                "Keyword search is better at the top spot, meaning search catches paraphrases further down. Together they beat both, and a reranker adds more at the top.")
    + T.section("02 / Honesty", 'Saying "I don\'t know"',
                T.callout("<b>This is the weak part.</b> At the chosen threshold the system answered "
                          f"{pct(a['answer_rate_on_answerable'])} of answerable questions but refused only {pct(a['refusal_rate_on_unanswerable'])} of the unanswerable ones. "
                          "The unanswerable questions in SQuAD 2.0 were written to look answerable, about the same topic as a real paragraph, so retrieval finds a very relevant "
                          "paragraph and the confidence stays high. A refusal signal based on search score alone is not enough here.", "warn")
                + T.card(abst_table),
                f"Threshold tuned on one half of the articles; every number comes from the other half ({r['corpus']['test_questions']:,} questions).")
    + T.section("03 / The answer itself", "Sentence quality",
                T.card(f'<div class="big">{pct(s["sentence_contains_gold_answer"])}</div><div class="cap">When the right paragraph ranks first, the sentence picked as the answer contains '
                       f'the correct answer text ({s["questions"]:,} questions). Answers are sentences copied from a document, so they cannot be invented.</div>'))
    + T.section("04 / Speed", "Per question, one laptop CPU",
                T.card(T.table(["Stage", "Average time"], lat_rows, numeric_cols=(1,))),
                "The reranker gives the best results but is by far the slowest step. Without it an answer takes about 10 ms.")
    + f'<div class="foot">Benchmark: SQuAD 2.0 dev, {r["corpus"]["chunks"]:,} paragraphs and {r["corpus"]["questions"]:,} sampled questions (half answerable, half not). '
      'Limits: one small sample, 1,204 paragraphs, English Wikipedia text only. Timings are per stage, not a load test.</div>')

html = T.page("Document Q&A evaluation", "Document Q&A", "Benchmark  ·  SQuAD 2.0 dev", "How well does it", "find the answer?",
              "Answers with sources, team permissions, and honest scoring. Every number below comes from a test the system did not train on.",
              body, plotly_js=get_plotlyjs(), accent="#4b3fd6", soft="#ecebff", soft_dark="#23223f", logo="Q")
open("outputs/report.html", "w", encoding="utf-8").write(html)
print("wrote outputs/report.html")
