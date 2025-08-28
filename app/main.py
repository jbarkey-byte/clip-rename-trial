import os, json, time, io, zipfile, uuid, mimetypes, hashlib, tempfile, subprocess, shutil, datetime
from typing import Dict
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response, JSONResponse
import boto3

AWS_REGION = os.environ.get("AWS_REGION", "us-east-2")
S3_BUCKET  = os.environ.get("S3_BUCKET")

s3 = boto3.client("s3", region_name=AWS_REGION)

app = FastAPI(title="Clip Rename Trial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Framer domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# demo in-memory store
JOBS: Dict[str, Dict] = {}

@app.get("/health")
def health():
    return {"ok": True}

def _run(cmd: list, check=True):
    """run a subprocess and return stdout text"""
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)
    return out.stdout.decode("utf-8", "ignore")

def _ffprobe_json(path: str):
    try:
        j = _run(["ffprobe","-v","error","-print_format","json","-show_format","-show_streams",path])
        return json.loads(j)
    except Exception:
        return {}

def _sha1_short(path: str, bytes_limit: int = 8_000_000):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        chunk = f.read(bytes_limit)  # hash first N bytes to keep it quick
        h.update(chunk)
    return h.hexdigest()[:8]

def _safe_slug(s: str):
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-").replace("--","-")

def _build_filename(scene, subject, action, date_str, shortid, ext):
    parts = [scene, subject, action, date_str, f"shortid-{shortid}"]
    return "--".join([_safe_slug(p) for p in parts if p]).strip("-") + f".{ext}"

def _write_metadata(path: str, title: str, keywords: list, scene: str):
    kw = ",".join(keywords)
    # Write XMP + QuickTime atoms with ExifTool (Spotlight & NLE friendly)
    cmd = [
        "exiftool", "-overwrite_original",
        f"-XMP-dc:Title={title}",
        f"-XMP-dc:Subject={kw}",
        f"-XMP-xmpDM:scene={scene}",
        f"-QuickTime:Title={title}",
        f"-QuickTime:com.apple.quicktime.keyword={kw}",
        path
    ]
    _run(cmd)

def _make_resolve_csv(path_csv: str, clip_name: str, scene: str, shot: str, take: str, keywords: list):
    # Minimal Resolve CSV
    import csv
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Clip Name","Scene","Shot","Take","Keywords","Comments"])
        w.writerow([clip_name, scene, shot, take, ";".join(keywords), "auto-generated"])

def _process_real(job_id: str, orig_name: str):
    """download file from S3, rename + embed metadata, zip result, upload to S3"""
    state = JOBS[job_id]

    def step(msg, pct=None):
        state["status"] = msg
        if pct is not None: state["pct"] = pct
        state["steps"].append(msg)

    step("Downloading clip…", 5)
    tmpdir = tempfile.mkdtemp(prefix="cliptrial-")
    try:
        input_key = state["object_key"]
        in_name = os.path.basename(input_key)
        local_in = os.path.join(tmpdir, in_name)
        s3.download_file(S3_BUCKET, input_key, local_in)

        step("Reading media info…", 15)
        info = _ffprobe_json(local_in)
        fmt = info.get("format", {})
        duration = float(fmt.get("duration", 0.0)) if fmt.get("duration") else 0.0
        # date: prefer container creation_time → else today
        date = None
        tags = (fmt.get("tags") or {})
        ct = tags.get("creation_time")
        if ct:
            try:
                date = datetime.datetime.fromisoformat(ct.replace("Z","+00:00")).date()
            except Exception:
                pass
        if not date:
            date = datetime.date.today()
        date_str = date.isoformat()

        step("Creating short clip ID…", 20)
        shortid = _sha1_short(local_in)

        # VERY SIMPLE labels for now (no heavy AI yet)
        # Try to guess scene/subject/action from original name; else use placeholders
        base, ext = os.path.splitext(in_name)
        ext = ext.lstrip(".").lower() or "mp4"
        tokens = [t for t in re_split(r"[-_ .]+", base) if t]
        # naive guesses
        scene  = (tokens[0] if tokens else "scene")
        subject= (tokens[1] if len(tokens)>1 else "subject")
        action = (tokens[2] if len(tokens)>2 else "action")

        step("Writing searchable metadata…", 40)
        title = f"{subject} {action} at {scene}"
        keywords = [scene, subject, action, f"shortid-{shortid}"]
        _write_metadata(local_in, title=title, keywords=keywords, scene=scene)

        step("Building new filename…", 55)
        new_name = _build_filename(scene, subject, action, date_str, shortid, ext)
        local_out = os.path.join(tmpdir, new_name)
        shutil.copy2(local_in, local_out)  # we already wrote metadata in-place; copy as the new name

        step("Preparing sidecars…", 65)
        resolve_csv = os.path.join(tmpdir, "Resolve.csv")
        _make_resolve_csv(resolve_csv, new_name, scene, "", "", keywords)

        step("Packaging ZIP…", 85)
        zip_path = os.path.join(tmpdir, "result.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(local_out, new_name)
            z.write(resolve_csv, "Resolve.csv")
            # tiny manifest
            z.writestr("manifest.json", json.dumps({
                "original": in_name,
                "renamed": new_name,
                "duration_sec": duration,
                "keywords": keywords
            }, indent=2))

        step("Uploading result…", 95)
        with open(zip_path, "rb") as f:
            s3.put_object(Bucket=S3_BUCKET, Key=state["result_key"], Body=f.read(), ContentType="application/zip")

        state["filename"] = new_name
        state["status"] = "Done"
        state["pct"] = 100
        state["ready"] = True
    finally:
        try: shutil.rmtree(tmpdir)
        except Exception: pass

# ------- routes -------

@app.post("/start_job")
async def start_job(req: Request):
    # tolerate empty or non-JSON body
    try:
        body = await req.json()
        if not isinstance(body, dict): body = {}
    except Exception:
        body = {}
    # tolerate form
    if not body:
        try:
            form = await req.form()
            body = dict(form)
        except Exception:
            body = {}

    filename = (body.get("filename") or "clip.mp4").strip()
    content_type = (body.get("content_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream")

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

@app.post("/finalize_upload")
async def finalize_upload(payload: Dict, background: BackgroundTasks):
    job_id = payload.get("job_id")
    orig_name = payload.get("name", "clip.mp4")
    if not job_id or job_id not in JOBS:
        raise HTTPException(status_code=400, detail="Unknown job_id")

    # (Optional) ensure object exists
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=JOBS[job_id]["object_key"])
    except Exception:
        raise HTTPException(status_code=400, detail="Upload not found in bucket")

    JOBS[job_id]["status"] = "Processing…"
    JOBS[job_id]["pct"] = 1
    background.add_task(_process_real, job_id, orig_name)
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

    # fallback: generate tiny ZIP in memory (shouldn't happen now)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "Zip not found in S3; fallback path.\n")
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename=\"clip-rename-trial-{job_id}.zip\""}
    return Response(content=buf.read(), media_type="application/zip", headers=headers)

# small util
import re
def re_split(pat, s): return [x for x in re.split(pat, s) if x]
