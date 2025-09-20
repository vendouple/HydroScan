import os
import threading
import time
import webbrowser
import gradio as gr
import httpx
from loguru import logger
from backend.config import settings

API_URL = f"http://{settings.api_host}:{settings.api_port}"


def start_backend():
    # Start uvicorn in a thread
    import uvicorn

    def run():
        uvicorn.run(
            "backend.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=False,
            log_level="info",
        )

    t = threading.Thread(target=run, daemon=True)
    t.start()
    # Wait until available
    for _ in range(60):
        try:
            r = httpx.get(f"{API_URL}/status", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)


def analyze_ui(images, videos, user_inputs, debug):
    files = []
    if images:
        files.extend(images)
    if videos:
        files.extend(videos)
    if not files:
        return {"error": "Provide at least one image or video"}

    m = httpx.MultipartWriter()
    if images:
        for f in images:
            with open(f, "rb") as fh:
                m.add_part(
                    fh,
                    headers={
                        "Content-Disposition": f"form-data; name=images; filename={os.path.basename(f)}"
                    },
                )
    if videos:
        for f in videos:
            with open(f, "rb") as fh:
                m.add_part(
                    fh,
                    headers={
                        "Content-Disposition": f"form-data; name=videos; filename={os.path.basename(f)}"
                    },
                )
    m.add_part(
        user_inputs or "",
        headers={"Content-Disposition": "form-data; name=user_inputs"},
    )
    m.add_part(
        debug or "none", headers={"Content-Disposition": "form-data; name=debug"}
    )

    r = httpx.post(
        f"{API_URL}/analyze",
        content=m,
        headers={"Content-Type": m.content_type},
        timeout=60,
    )
    r.raise_for_status()
    job = r.json()
    job_id = job["id"]
    # poll results
    for _ in range(240):
        time.sleep(1.0)
        rr = httpx.get(f"{API_URL}/results/{job_id}")
        if rr.status_code == 200:
            res = rr.json()
            lines = [
                f"Scene: {res['scene']}",
                f"Potability: {res['potability']:.1f}%",
                f"Confidence: {res['confidence']:.1f}%",
                "",
                "Timeline:",
            ]
            for step in res.get("timeline", []):
                lines.append(str(step))
            return "\n".join(lines)
    return "Timed out waiting for results."


def build_ui():
    with gr.Blocks(title="HydroScan") as demo:
        gr.Markdown("# HydroScan — Water Impurity Analysis (Prototype)")
        with gr.Row():
            with gr.Column():
                image_in = gr.Files(
                    label="Upload images",
                    file_types=[".jpg", ".jpeg", ".png", ".webp", ".bmp"],
                    type="filepath",
                )
                video_in = gr.Files(
                    label="Upload videos (≤60s)",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"],
                    type="filepath",
                )
                user_inputs = gr.Textbox(label="Tell us about the water (optional)")
                debug = gr.Dropdown(
                    ["none", "minimal", "detailed"], value="none", label="Debug level"
                )
                analyze_btn = gr.Button("Analyze")
            with gr.Column():
                out_txt = gr.Textbox(label="Results", lines=20)
        analyze_btn.click(
            analyze_ui, inputs=[image_in, video_in, user_inputs, debug], outputs=out_txt
        )
    return demo


if __name__ == "__main__":
    start_backend()
    ui = build_ui()
    ui.launch(server_name=settings.api_host, server_port=settings.ui_port)
