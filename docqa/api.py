"""A small web API and page for asking questions.

  uvicorn docqa.api:app --port 8001      then open http://localhost:8001

Users and their teams are hard-coded for the demo. A real system would read the user's groups from the login token.
"""
from __future__ import annotations
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import corpus
from .qa import QA
from .search import Index, ROOT

USERS = {"alice": ["engineering"], "bob": ["finance"], "carol": ["hr"], "dave": ["engineering", "finance"], "admin": None}
app = FastAPI(title="Document Q&A")
_qa: QA | None = None


def get_qa() -> QA:
    global _qa
    if _qa is None:
        chunks, _ = corpus.load()
        threshold = 0.0
        res = os.path.join(ROOT, "outputs", "results.json")
        if os.path.exists(res):
            threshold = json.load(open(res))["abstention_test"]["threshold"]
        _qa = QA(Index(chunks), threshold)
    return _qa


class Ask(BaseModel):
    question: str
    user: str = "admin"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Ask):
    if body.user not in USERS:
        raise HTTPException(404, f"unknown user, try one of {list(USERS)}")
    if not body.question.strip():
        raise HTTPException(422, "question is empty")
    a = get_qa().ask(body.question, USERS[body.user])
    return dict(abstained=a.abstained, confidence=round(a.confidence, 3), answer=a.answer, message=a.message,
                citations=a.citations, user=body.user, visible_teams=USERS[body.user] or "all")


@app.get("/", response_class=HTMLResponse)
def page():
    return """<!doctype html><meta charset=utf-8><title>Document Q&A</title>
<style>body{font:16px/1.5 Segoe UI,Arial;max-width:760px;margin:30px auto;padding:0 14px}input,select,button{font:inherit;padding:8px}
input{width:62%}.c{border:1px solid #ddd;border-radius:8px;padding:10px;margin:8px 0;font-size:14px}.i{color:#b45309}</style>
<h2>Document Q&A</h2><p>Ask about the documents your team can see. Answers are sentences copied from a document, with sources.</p>
<select id=u><option>admin<option>alice<option>bob<option>carol<option>dave</select>
<input id=q placeholder="Ask a question"> <button onclick=go()>Ask</button><div id=o></div>
<script>async function go(){const r=await fetch('/ask',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({question:q.value,user:u.value})});const d=await r.json();
let h=d.abstained?`<p class=i><b>${d.message}</b> (confidence ${d.confidence})</p>`:`<p><b>${d.answer}</b> (confidence ${d.confidence})</p>`;
(d.citations||[]).forEach((c,i)=>h+=`<div class=c><b>[${i+1}] ${c.title}</b> (${c.team})<br>${c.snippet}</div>`);o.innerHTML=h}</script>"""
