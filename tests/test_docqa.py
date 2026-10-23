import os, sys, random
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docqa import corpus
from docqa.qa import QA, build_prompt, generate, split_sentences
from docqa.search import Index, rrf, tokenize


@pytest.fixture(scope="module")
def setup():
    chunks, questions = corpus.load()
    return Index(chunks, use_dense=True, use_rerank=False), questions


def test_rrf_rewards_items_ranked_high_in_both_lists():
    fused = [i for i, _ in rrf([[1, 2, 3], [3, 1, 4]])]
    assert fused[0] == 1 and set(fused) == {1, 2, 3, 4}


def test_tokenize_drops_stop_words_and_punctuation():
    assert tokenize("What is the Capital of France?") == ["capital", "france"]


def test_sentence_splitter():
    assert len(split_sentences("It rained. Then it stopped. The end.")) == 3


def test_bm25_finds_an_exact_term(setup):
    index, _ = setup
    mask = index.allowed_mask(None)
    ids, _ = index.bm25_search("Normans Rollo Normandy", mask, 3)
    assert any("Norman" in index.chunks[i].text for i in ids)


def test_permissions_never_leak(setup):
    """Property test: whichever team a user belongs to, no search method may return a chunk from another team."""
    index, questions = setup
    rng = random.Random(0)
    for _ in range(60):
        q = rng.choice(questions)
        groups = rng.choice([["engineering"], ["finance"], ["hr"], ["engineering", "hr"]])
        mask = index.allowed_mask(groups)
        qv = index.embed_queries([q.text])[0]
        for ids in (index.bm25_search(q.text, mask, 20)[0], index.dense_search(qv, mask, 20)[0],
                    index.hybrid_search(q.text, qv, mask, 20)[0]):
            assert all(index.chunks[i].team in groups for i in ids)


def test_user_with_no_access_gets_a_clear_refusal(setup):
    index, questions = setup
    a = QA(index, threshold=0.0).ask(questions[0].text, groups=[])
    assert a.abstained and "access" in a.message


def test_gibberish_is_refused_with_a_high_threshold(setup):
    index, _ = setup
    qa = QA(index, threshold=0.75)
    assert qa.ask("zxqv blorf wibble quux", None).abstained


def test_answer_comes_with_a_citation_from_an_allowed_team(setup):
    index, questions = setup
    q = next(q for q in questions if q.answerable)
    team = index.chunks[q.chunk_id].team
    a = QA(index, threshold=0.0).ask(q.text, [team])
    assert not a.abstained and a.citations and all(c["team"] == team for c in a.citations)


def test_llm_prompt_contains_only_allowed_passages(setup):
    """If a language model is used, it must never even see text the user is not allowed to read."""
    index, questions = setup
    q = next(q for q in questions if q.answerable)
    allowed_team = index.chunks[q.chunk_id].team
    seen = {}

    class Spy:
        def complete(self, system, user):
            seen["prompt"] = user
            return "ok [1]"

    generate(QA(index, threshold=0.0), q.text, [allowed_team], Spy())
    forbidden = [c.text[:80] for c in index.chunks if c.team != allowed_team][:300]
    assert "prompt" in seen and not any(f in seen["prompt"] for f in forbidden)


def test_build_prompt_numbers_passages(setup):
    index, _ = setup
    system, user = build_prompt("why?", index.chunks[:2])
    assert "[1]" in user and "[2]" in user and "I don't know" in system
