from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when running as a script
if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


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
        from WebInterface.backend.Utils.fetch_models import ensure_models

        ensure_models(MODELS_DIR)
    except Exception as e:
        # Non-fatal: continue to boot; adapters should handle missing assets with clear errors
        print(f"[HydroScan] Model fetcher skipped or failed: {e}")

    try:
        from WebInterface.backend.Adapters.RFDETR import RFDETRAdapter
        from WebInterface.backend.Adapters.InModel import InModelAdapter
        from WebInterface.backend.Adapters.Place365 import Place365Adapter

        _ = RFDETRAdapter()
        _ = InModelAdapter(models_dir=MODELS_DIR)
        _ = Place365Adapter(models_dir=MODELS_DIR)
        print("[HydroScan] Models initialized (RT-DETR, InModel, Places365)")
    except Exception as e:
        print(f"[HydroScan] Adapter warm-up skipped: {e}")

    # Mount static files
    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    templates = Jinja2Templates(directory=TEMPLATES_DIR)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse("home.html", {"request": request})

    @app.get("/app", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/model-lab", response_class=HTMLResponse)
    def model_lab(request: Request):
        return templates.TemplateResponse("model-lab.html", {"request": request})

    # Include core API routers independently so optional feature failures do not
    # disable the entire API surface.
    core_routers = (
        ("analyze", "/api"),
        ("results", "/api"),
        ("models", "/api"),
        ("model_test", "/api"),
    )
    for module_name, prefix in core_routers:
        try:
            module = __import__(f"WebInterface.API.{module_name}", fromlist=["router"])
            app.include_router(module.router, prefix=prefix)
        except Exception as e:
            print(f"[HydroScan] Failed to include router '{module_name}': {e}")

    # Akinator is optional and owns its own route prefix.
    try:
        from WebInterface.API import akinator

        app.include_router(akinator.router)
    except Exception as e:
        print(f"[HydroScan] Optional akinator router unavailable: {e}")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)
