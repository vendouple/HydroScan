from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os


APP_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(APP_DIR, "frontend")
TEMPLATES_DIR = os.path.join(FRONTEND_DIR, "templates")
STATIC_DIR = os.path.join(FRONTEND_DIR, "static")
BACKEND_DIR = os.path.join(APP_DIR, "backend")
MODELS_DIR = os.path.join(BACKEND_DIR, "Models")


def create_app() -> FastAPI:
    app = FastAPI(title="HydroScan")

    # On boot, attempt to ensure required model files are present
    try:
        from .backend.Utils.fetch_models import ensure_models  # type: ignore

        ensure_models(MODELS_DIR)
    except Exception as e:
        # Non-fatal: continue to boot; adapters should handle missing assets with clear errors
        print(f"[HydroScan] Model fetcher skipped or failed: {e}")

    # Mount static files
    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    # Include API routers if present
    try:
        from .API import analyze, results  # noqa: F401

        app.include_router(analyze.router, prefix="/api")
        app.include_router(results.router, prefix="/api")
    except Exception:
        # Allow app to boot even if routers are not ready yet
        pass

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("WebInterface.app:app", host=host, port=port, reload=True)
