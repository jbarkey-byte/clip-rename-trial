# app/main.py
import os, json, time, io, zipfile, uuid, mimetypes
from typing import Dict
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response, JSONResponse
import boto3  # <-- this is the important import

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
S3_BUCKET = os.environ.get("S3_BUCKET")

s3 = boto3.client("s3", region_name=AWS_REGION)

app = FastAPI(title="Clip Rename Trial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later to your Framer domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory jobs store for demo
JOBS: Dict[str, Dict] = {}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start_job")
async def start_job(req: Request):
    """
    Client sends: { filename: 'clip.mp4', content_type: 'video/mp4' }
    We return { job_id, upload_url (S3 presigned PUT), object_key }
    """
    # Be tolerant: body might be empty or not JSON
    try:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    # Also tolerate form submissions
    if not body:
        try:
            form = await req.form()
            body = dict(form)
        except Exception:
            body = {}

    filename = (body.get("filename") or "clip.mp4").strip()
    import mimetypes
    content_type = (body.get("content_type")
                    or mimetypes.guess_type(filename)[0]
                    or "application/octet-stream")

    import uuid
    job_id = str(uuid.uuid4())
    object_key = f"uploads/{job_id}/{filename}"

    try:
        upload_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": S3_BUCKET, "Key": object_key, "ContentType": content_type},
            ExpiresIn=3600,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Presign error: {e}")

    JOBS[job_id] = {
        "status": "waiting for upload",
        "pct": 0,
        "steps": [],
        "ready": False,
        "filename": None,
        "object_key": object_key,
        "result_key": f"outputs/{job_id}/result.zip",
    }
    return {"job_id": job_id, "upload_url": upload_url, "object_key": object_key}

def _simulate_processing(job_id: str, orig_name: str):
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
        time.sleep(0.9)

    JOBS[job_id]["filename"] = "skatepark--skateboarder--kickflip--2025-08-27--shortid-7c9ad1ef.mp4"
    JOBS[job_id]["status"] = "Done"
    JOBS[job_id]["pct"] = 100
    JOBS[job_id]["ready"] = True

    # Write a small ZIP to S3 so /download can hand you a signed link
    try:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("README.txt", "Your processed clip would be here in the real build.\n")
            z.writestr("proposed_filename.txt", JOBS[job_id]["filename"] or "unknown.mp4")
        buf.seek(0)
        s3.put_object(Bucket=S3_BUCKET, Key=JOBS[job_id]["result_key"],
                      Body=buf.getvalue(), ContentType="application/zip")
    except Exception:
        pass  # ok if this fails in demo

@app.post("/finalize_upload")
async def finalize_upload(payload: Dict, background: BackgroundTasks):
    job_id = payload.get("job_id")
    orig_name = payload.get("name", "clip.mp4")
    if not job_id or job_id not in JOBS:
        raise HTTPException(status_code=400, detail="Unknown job_id")

    # (Optional) verify upload exists: s3.head_object(Bucket=S3_BUCKET, Key=JOBS[job_id]["object_key"])
    JOBS[job_id]["status"] = "Processing…"
    JOBS[job_id]["pct"] = 1
    background.add_task(_simulate_processing, job_id, orig_name)
    return {"ok": True, "job_id": job_id}

@app.get("/progress/{job_id}")
def progress(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    def event_stream():
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

    result_key = JOBS[job_id].get("result_key")
    if result_key:
        try:
            signed = s3.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": S3_BUCKET, "Key": result_key},
                ExpiresIn=600,
            )
            return JSONResponse({"redirect": signed})
        except Exception:
            pass

    # Fallback: serve tiny ZIP directly
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "Your processed clip would be here in the real build.\n")
        z.writestr("proposed_filename.txt", JOBS[job_id]["filename"] or "unknown.mp4")
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="clip-rename-trial-{job_id}.zip"'}
    return Response(content=buf.read(), media_type="application/zip", headers=headers)
