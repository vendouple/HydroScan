import pytest
import asyncio
import httpx
import threading
import time

# Start backend once for tests
started = False


def ensure_backend():
    global started
    if started:
        return
    import uvicorn

    def run():
        uvicorn.run(
            "backend.main:app",
            host="127.0.0.1",
            port=8001,
            reload=False,
            log_level="warning",
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()
    for _ in range(60):
        try:
            r = httpx.get("http://127.0.0.1:8001/status", timeout=1.0)
            if r.status_code == 200:
                started = True
                return
        except Exception:
            pass
        time.sleep(0.5)


def test_status_endpoint():
    ensure_backend()
    r = httpx.get("http://127.0.0.1:8001/status", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "ok" in data and data["ok"] is True
