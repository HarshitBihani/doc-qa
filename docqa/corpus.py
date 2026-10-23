"""Load the document collection and the evaluation questions.

Data: SQuAD 2.0 dev set (Wikipedia articles with questions written by people). Every paragraph is one "document chunk".
Half of the questions are answerable from their paragraph. The other half look answerable but are NOT: those are
what we use to test whether the system refuses instead of bluffing.

Access control demo: each article is assigned to one of three teams. The assignment is arbitrary (SQuAD has no real
permissions); it exists so the permission filter can be tested properly.
"""
from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "dev-v2.0.json")
TEAMS = ["engineering", "finance", "hr"]


@dataclass
class Chunk:
    chunk_id: int
    title: str
    text: str
    team: str            # which team's documents this belongs to


@dataclass
class Question:
    qid: str
    text: str
    chunk_id: int        # the paragraph the question was written about
    answerable: bool
    answers: list[str]
    title: str


def load():
    data = json.load(open(PATH, encoding="utf-8"))["data"]
    chunks, questions = [], []
    for ai, article in enumerate(data):
        team = TEAMS[ai % len(TEAMS)]
        for para in article["paragraphs"]:
            cid = len(chunks)
            chunks.append(Chunk(cid, article["title"], para["context"], team))
            for qa in para["qas"]:
                answers = sorted({a["text"] for a in qa.get("answers", [])})
                questions.append(Question(qa["id"], qa["question"], cid, not qa["is_impossible"], answers, article["title"]))
    return chunks, questions


def sample_questions(questions, n_each: int = 500, seed: int = 7):
    """A fixed, balanced sample: n_each answerable and n_each unanswerable questions."""
    rng = random.Random(seed)
    ans = [q for q in questions if q.answerable]
    una = [q for q in questions if not q.answerable]
    return rng.sample(ans, n_each) + rng.sample(una, n_each)


def split_by_article(questions):
    """Dev/test split by ARTICLE, so no article appears in both. Thresholds are tuned on dev and reported on test."""
    titles = sorted({q.title for q in questions})
    dev_titles = {t for i, t in enumerate(titles) if i % 2 == 0}
    dev = [q for q in questions if q.title in dev_titles]
    test = [q for q in questions if q.title not in dev_titles]
    return dev, test
