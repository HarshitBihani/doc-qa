import os, sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from docqa import api, corpus
from docqa.qa import QA
from docqa.search import Index


@pytest.fixture(scope="module")
def client():
    chunks, questions = corpus.load()
    api._qa = QA(Index(chunks, use_dense=True, use_rerank=False), threshold=0.0)    # reranker off: fast tests
    return TestClient(api.app), chunks, questions


def test_health_and_page(client):
    c, _, _ = client
    assert c.get("/health").json() == {"status": "ok"} and "Document Q&A" in c.get("/").text


def test_answers_only_cite_the_users_team(client):
    c, chunks, questions = client
    q = next(q for q in questions if q.answerable and chunks[q.chunk_id].team == "engineering")
    r = c.post("/ask", json={"question": q.text, "user": "alice"}).json()
    assert r["citations"] and all(x["team"] == "engineering" for x in r["citations"])


def test_user_outside_the_team_cannot_get_that_document(client):
    c, chunks, questions = client
    q = next(q for q in questions if q.answerable and chunks[q.chunk_id].team == "hr")
    r = c.post("/ask", json={"question": q.text, "user": "bob"}).json()          # bob is in finance
    assert all(x["team"] == "finance" for x in r["citations"])
    assert all(x["chunk_id"] != q.chunk_id for x in r["citations"])


def test_bad_requests(client):
    c, _, _ = client
    assert c.post("/ask", json={"question": "hi", "user": "mallory"}).status_code == 404
    assert c.post("/ask", json={"question": "  ", "user": "admin"}).status_code == 422
