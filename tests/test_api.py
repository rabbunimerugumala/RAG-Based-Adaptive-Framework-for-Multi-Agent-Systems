"""HTTP API tests with FastAPI's TestClient (dev-mode models)."""

import time

import pytest
from fastapi.testclient import TestClient

from adaptive_rag.api.app import create_app
from tests.conftest import dev_bundle, make_pdf

DOC = b"The Quillon protocol was created by Mara Ostberg in 2011. Quillon servers listen on TCP port 7425."


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, dev_bundle())) as c:
        yield c


def upload(client, name, data):
    return client.post("/api/documents", files=[("files", (name, data, "application/octet-stream"))])


def test_health_reports_mode_models_and_kb(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["mode"] == "dev" and "Development" in h["dev_mode_warning"]
    assert set(h["models"]) == {"embedder", "generator", "nli", "reranker"}
    assert h["knowledge_base"]["documents"] == 0 and h["config"]["max_retries"] == 2


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/health", "/api/documents", "/api/query", "/api/compare", "/api/evaluations"):
        assert p in paths


def test_upload_list_query_delete_flow(client):
    r = upload(client, "quillon.txt", DOC)
    assert r.status_code == 200 and r.json()[0]["status"] == "indexed"
    doc_id = r.json()[0]["document"]["doc_id"]
    docs = client.get("/api/documents").json()
    assert len(docs) == 1 and docs[0]["filename"] == "quillon.txt" and docs[0]["file_type"] == "txt"

    q = client.post("/api/query", json={"question": "Which TCP port do Quillon servers listen on?", "method": "standard"})
    body = q.json()
    assert q.status_code == 200 and body["status"] == "answered" and "7425" in body["answer"]
    assert body["citations"][0]["filename"] == "quillon.txt" and body["citations"][0]["chunk_index"] == 0

    assert client.get(f"/api/runs/{body['run_id']}").json()["answer"] == body["answer"]
    assert client.get("/api/runs").json()[0]["run_id"] == body["run_id"]

    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    empty = client.post("/api/query", json={"question": "Which port?", "method": "adaptive"}).json()
    assert empty["status"] == "no_documents"


def test_upload_validation_errors(client, settings):
    res = client.post("/api/documents", files=[
        ("files", ("bad.exe", b"MZ", "application/octet-stream")),
        ("files", ("big.txt", b"a" * (settings.max_upload_mb * 1024 * 1024 + 1), "text/plain")),
        ("files", ("scan.pdf", make_pdf([""]), "application/pdf")),
        ("files", ("ok.md", b"# Notes\nOtters hold hands.", "text/markdown")),
        ("files", ("dup.md", b"# Notes\nOtters hold hands.", "text/markdown")),
    ]).json()
    assert [r["status"] for r in res] == ["failed", "failed", "failed", "indexed", "duplicate"]
    assert "Unsupported" in res[0]["message"] and "MB" in res[1]["message"] and "OCR" in res[2]["message"]
    assert client.post("/api/documents").status_code == 422


def test_query_request_validation(client):
    assert client.post("/api/query", json={"question": ""}).status_code == 422
    assert client.post("/api/query", json={"question": "valid?", "method": "magic"}).status_code == 422
    assert client.get("/api/runs/doesnotexist").status_code == 404
    assert client.get("/api/runs/..%2F..%2Fsecret").status_code == 404


def test_compare_runs_both_methods_on_same_kb(client):
    upload(client, "quillon.txt", DOC)
    r = client.post("/api/compare", json={"question": "Who created the Quillon protocol?"}).json()
    assert r["standard"]["method"] == "standard" and r["adaptive"]["method"] == "adaptive"
    assert r["standard"]["verification_used_in_decision"] is False
    assert r["adaptive"]["verification_used_in_decision"] is True
    assert {t["step"] for t in r["adaptive"]["trace"]} >= {"planner", "evidence", "verifier", "confidence"}


def test_evaluation_lifecycle_and_export(client):
    datasets = client.get("/api/datasets").json()
    smoke = next(d for d in datasets if d["name"] == "smoke_test")
    assert smoke["synthetic"] is True and smoke["num_questions"] == 24
    assert client.post("/api/evaluations", json={"dataset": "nope"}).status_code == 404
    job = client.post("/api/evaluations", json={"dataset": "smoke_test", "limit": 4}).json()
    for _ in range(100):
        state = client.get(f"/api/evaluations/{job['job_id']}").json()
        if state["status"] != "running":
            break
        time.sleep(0.1)
    assert state["status"] == "completed" and state["done"] == state["total"] == 8
    assert set(state["result"]["summary"]) == {"standard", "adaptive"}
    csv = client.get(f"/api/evaluations/{job['job_id']}/export?format=csv")
    assert csv.status_code == 200 and csv.text.startswith("question_id")
    assert client.get(f"/api/evaluations/{job['job_id']}/export?format=json").json()["run_id"] == job["job_id"]
    assert client.get(f"/api/evaluations/{job['job_id']}/export?format=xml").status_code == 422
    assert any(j["job_id"] == job["job_id"] for j in client.get("/api/evaluations").json())
    assert client.get("/api/evaluations/..evil").status_code == 400
