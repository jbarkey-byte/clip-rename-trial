# app/main.py
import json, time, threading, io, zipfile, uuid
from typing import Dict
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response

app = FastAPI(title="Clip Rename Trial")

# Allow calls from your Framer site or anywhere while testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory "jobs" store for demo
JOBS: Dict[str, Dict] = {}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start_job")
def start_job():
    job_id = str(uuid.uuid4())
    # In the real app, we'd return a signed upload URL here.
    JOBS[job_id] = {
        "status": "waiting for upload",
        "pct": 0,
        "steps": [],
        "ready": False,
        "filename": None,
    }
    # Placeholder "upload URL" (not used yet)
    return {"job_id": job_id, "upload_url": "https://example.com/placeholder-upload"}

def _simulate_processing(job_id: str, orig_name: str):
    # Fake stages to prove progress & SSE work
    stages = [
        "Analyzing audio…",
        "Detecting scenes…",
        "Looking for a slate at the start…",
        "Looking for a slate at the end…",
        "Scanning the whole clip for a slate…",
        "Figuring out the main subject and action…",
        "Building your filename…",
        "Writing searchable metadata…",
        "Preparing your download…",
    ]
    total = len(stages)
    for i, stage in enumerate(stages, start=1):
        JOBS[job_id]["status"] = stage
        JOBS[job_id]["pct"] = int(i * 100 / total)
        JOBS[job_id]["steps"].append(stage)
        time.sleep(1.0)  # simulate work
    # Pretend we computed a new filename
    JOBS[job_id]["filename"] = "skatepark--skateboarder--kickflip--2025-08-27--shortid-7c9ad1ef.mp4"
    JOBS[job_id]["status"] = "Done"
    JOBS[job_id]["pct"] = 100
    JOBS[job_id]["ready"] = True

@app.post("/finalize_upload")
def finalize_upload(payload: Dict, background: BackgroundTasks):
    job_id = payload.get("job_id")
    orig_name = payload.get("name", "clip.mp4")
    if not job_id or job_id not in JOBS:
        raise HTTPException(status_code=400, detail="Unknown job_id")
    JOBS[job_id]["status"] = "Processing…"
    JOBS[job_id]["pct"] = 1
    # Kick off background processing
    background.add_task(_simulate_processing, job_id, orig_name)
    return {"ok": True, "job_id": job_id}

@app.get("/progress/{job_id}")
def progress(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    def event_stream():
        # Stream current state every 1s until ready
        while True:
            state = JOBS.get(job_id)
            if not state:
                break
            data = {
                "status": state["status"],
                "pct": state["pct"],
                "ready": state["ready"],
                "filename": state["filename"],
            }
            yield f"data: {json.dumps(data)}\n\n"
            if state["ready"]:
                break
            time.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/download/{job_id}")
def download(job_id: str):
    if job_id not in JOBS or not JOBS[job_id]["ready"]:
        raise HTTPException(status_code=404, detail="Not ready or unknown job")

    # Create a tiny in-memory ZIP as a placeholder (we'll replace with real outputs later)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "Your processed clip would be here in the real build.\n")
        z.writestr("proposed_filename.txt", JOBS[job_id]["filename"] or "unknown.mp4")
    buf.seek(0)

    headers = {
        "Content-Disposition": f'attachment; filename="clip-rename-trial-{job_id}.zip"'
    }
    return Response(content=buf.read(), media_type="application/zip", headers=headers)
