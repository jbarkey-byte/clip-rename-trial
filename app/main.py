# app/main.py
import os
import io
import re
import cv2
import json
import time
import uuid
import math
import shutil
import boto3
import zipfile
import hashlib
import tempfile
import mimetypes
import subprocess
import datetime
import numpy as np
from typing import Dict, List, Tuple
from collections import Counter

import requests
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import (
    StreamingResponse,
    Response,
    JSONResponse,
    RedirectResponse,
)

# ---------- ENV / GLOBALS ----------
AWS_REGION     = os.environ.get("AWS_REGION", "us-east-2")
S3_BUCKET      = os.environ.get("S3_BUCKET")
SECRET_KEY     = os.environ.get("SECRET_KEY", "dev-secret-change-me")
POSTMARK_TOKEN = os.environ.get("POSTMARK_TOKEN")
APP_BASE_URL   = os.environ.get("APP_BASE_URL", "https://clip-rename-trial.onrender.com")
FROM_EMAIL     = os.environ.get("FROM_EMAIL", "no-reply@example.com")  # verified in Postmark

WHISPER_MODEL   = os.environ.get("WHISPER_MODEL", "tiny")   # tiny/base for CPU
WHISPER_MAX_SEC = int(os.environ.get("WHISPER_MAX_SEC", "120"))
CPU_THREADS     = int(os.environ.get("CPU_THREADS", "4"))

s3 = boto3.client("s3", region_name=AWS_REGION)
signer = URLSafeTimedSerializer(SECRET_KEY)

# lazy-loaded models
_nlp = None
_whisper = None

app = FastAPI(title="Clip Rename Trial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (OK for the trial)
JOBS: Dict[str, Dict] = {}

# ---------- small utils ----------

def _run(cmd: list, check=True) -> str:
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)
    return out.stdout.decode("utf-8", "ignore")

def _ffprobe_json(path: str) -> dict:
    try:
        j = _run([
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", path
        ])
        return json.loads(j)
    except Exception:
        return {}

def _has_audio_stream(ffprobe_info: dict) -> bool:
    streams = ffprobe_info.get("streams") or []
    for s in streams:
        if s.get("codec_type") == "audio":
            ch = int(str(s.get("channels", "0")) or "0")
            sr = int(str(s.get("sample_rate", "0")) or "0")
            if ch > 0 and sr > 0:
                return True
    return False

def _sha1_short(path: str, bytes_limit: int = 8_000_000) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(bytes_limit))
    return h.hexdigest()[:8]

def _safe_slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in s).strip("-").replace("--", "-")

def _truncate_token(s: str, maxlen: int = 28) -> str:
    s = s[:maxlen]
    return re.sub(r"-{2,}", "-", s).strip("-")

def _build_filename(scene, subject, action, date_str, shortid, ext) -> str:
    parts = [scene, subject, action, date_str, f"shortid-{shortid}"]
    parts = [_truncate_token(_safe_slug(p)) for p in parts if p]
    return "--".join([p for p in parts if p]) + f".{ext}"

def _write_metadata(path: str, title: str, keywords: list, scene: str):
    # No-op for now (ExifTool not in this build)
    return

def _make_resolve_csv(path_csv: str, clip_name: str, scene: str, shot: str, take: str, keywords: list):
    import csv
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Clip Name","Scene","Shot","Take","Keywords","Comments"])
        w.writerow([clip_name, scene, shot, take, ";".join(keywords), "auto-generated"])

def re_split(pat: str, s: str):
    return [x for x in re.split(pat, s) if x]

def send_magic_email(to_email: str, link_url: str):
    if not POSTMARK_TOKEN:
        raise RuntimeError("POSTMARK_TOKEN not set")
    r = requests.post(
        "https://api.postmarkapp.com/email",
        headers={
            "X-Postmark-Server-Token": POSTMARK_TOKEN,
            "Accept": "application/json",
        },
        json={
            "From": FROM_EMAIL,
            "To": to_email,
            "Subject": "Your download is ready",
            "TextBody": f"Your clip is ready. This link is valid for 24 hours:\n\n{link_url}\n",
        },
        timeout=15,
    )
    if not r.ok:
        try:
            detail = r.json().get("Message")
        except Exception:
            detail = r.text
        raise RuntimeError(f"Postmark {r.status_code}: {detail}")

# ---------- AI helpers (Whisper + spaCy) ----------

def _load_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        _nlp.add_pipe("sentencizer")
    return _nlp

def _load_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(
            WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            cpu_threads=CPU_THREADS
        )
    return _whisper

def _extract_audio(src_video: str, out_wav: str, max_seconds: int = 120):
    cmd = [
        "ffmpeg", "-y", "-i", src_video,
        "-vn", "-ac", "1", "-ar", "16000",
        "-t", str(max_seconds),
        out_wav
    ]
    _run(cmd)

def _transcribe(audio_path: str) -> str:
    model = _load_whisper()
    segments, _ = model.transcribe(
        audio_path,
        language="en",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=1,
        best_of=1,
        temperature=0.0,
        no_speech_threshold=0.3
    )
    out = []
    for seg in segments:
        if seg.no_speech_prob and seg.no_speech_prob > 0.6:
            continue
        out.append(seg.text.strip())
        if sum(len(x) for x in out) > 4000:
            break
    return " ".join(out).strip()

def _most_common(items: List[str]) -> str:
    items = [x for x in items if x]
    return Counter(items).most_common(1)[0][0] if items else ""

def _labels_from_transcript(text: str) -> Tuple[str, str, str]:
    """(scene, subject, action) from transcript text."""
    if not text or len(text) < 5:
        return ("", "", "")

    nlp = _load_nlp()
    doc = nlp(text)

    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    subject = _most_common(persons)
    if not subject:
        proper = [t.text for t in doc if t.pos_ == "PROPN" and len(t.text) > 1]
        subject = _most_common(proper)
    if not subject:
        noun_chunks = [nc.root.text for nc in doc.noun_chunks]
        subject = _most_common(noun_chunks)

    verbs = [t.lemma_ for t in doc if t.pos_ == "VERB" and t.lemma_ not in {"be","do","have"}]
    action = _most_common(verbs)

    scene = ""
    m = re.search(r"\b(at|in|on|near|inside|outside)\s+(the\s+|a\s+|an\s+)?([A-Za-z0-9\- ]{3,40})", text, flags=re.IGNORECASE)
    if m:
        scene = m.group(3).strip()
    if not scene:
        locs = [ent.text for ent in doc.ents if ent.label_ in {"GPE","LOC","FAC"}]
        scene = _most_common(locs)

    def clean(s: str) -> str:
        s = re.sub(r"\s+", " ", s or "")
        return s.strip()

    return (clean(scene), clean(subject), clean(action))

# ---------- Visual cues (CPU, OpenCV) ----------

def _color_ratios(bgr_img: np.ndarray) -> Tuple[float, float, float, float]:
    """Return approx ratios for green, blue, yellow, gray."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # green
    mask_g = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # blue (sky/water)
    mask_b = cv2.inRange(hsv, (90, 40, 40), (130, 255, 255))
    # yellow/sand
    mask_y = cv2.inRange(hsv, (20, 40, 40), (35, 255, 255))
    # low saturation = gray/indoor
    mask_gray = cv2.inRange(hsv, (0, 0, 40), (179, 40, 220))
    total = bgr_img.shape[0] * bgr_img.shape[1]
    eps = 1e-6
    return (
        float(mask_g.sum()) / 255.0 / (total + eps),
        float(mask_b.sum()) / 255.0 / (total + eps),
        float(mask_y.sum()) / 255.0 / (total + eps),
        float(mask_gray.sum()) / 255.0 / (total + eps),
    )

_PEOPLE_HOG = None
_FACE_CASCADE = None

def _load_detectors():
    global _PEOPLE_HOG, _FACE_CASCADE
    if _PEOPLE_HOG is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _PEOPLE_HOG = hog
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _detect_people_and_faces(bgr_img: np.ndarray) -> Tuple[int, int]:
    _load_detectors()
    # people (HOG) on smaller image for speed
    h, w = bgr_img.shape[:2]
    target_w = 640
    if w > target_w:
        img_small = cv2.resize(bgr_img, (target_w, int(h * target_w / max(1, w))))
    else:
        img_small = bgr_img
    rects, _ = _PEOPLE_HOG.detectMultiScale(img_small, winStride=(8,8), padding=(8,8), scale=1.05)
    people = len(rects)
    # faces (cascade)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48))
    return people, len(faces)

def _motion_magnitude(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=2, winsize=15,
                                        iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=False)
    return float(np.mean(mag))

def _sample_frames(video_path: str, max_frames: int = 60) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = frame_count / max(1.0, fps)
    # FIXED: balanced step computation without extra parens
    step = max(1, int(frame_count / max(1, min(max_frames, int(dur) + 1))))
    frames = []
    idx = 0
    while idx < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if len(frames) >= max_frames:
            break
        idx += step
    cap.release()
    return frames

def _visual_labels(video_path: str) -> Tuple[str, str, str, dict]:
    """
    Returns (scene, subject, action, debug_stats) from visuals.
    Heuristics: color ratios, people/face counts, motion magnitude.
    """
    frames = _sample_frames(video_path, max_frames=40)
    if not frames:
        return ("scene", "subject", "action", {"frames": 0})

    totals = dict(g=0.0,b=0.0,y=0.0,gray=0.0, people=0, faces=0, motion=0.0, frames=len(frames))
    prev_gray = None
    for f in frames:
        h, w = f.shape[:2]
        # resize for speed
        scale = 480.0 / max(h, w)
        if scale < 1.0:
            f = cv2.resize(f, (int(w*scale), int(h*scale)))
        g, b, y, gr = _color_ratios(f)
        p, fa = _detect_people_and_faces(f)
        totals["g"] += g; totals["b"] += b; totals["y"] += y; totals["gray"] += gr
        totals["people"] += p; totals["faces"] += fa
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            totals["motion"] += _motion_magnitude(prev_gray, gray)
        prev_gray = gray

    n = max(1, len(frames))
    avg_g   = totals["g"]/n
    avg_b   = totals["b"]/n
    avg_y   = totals["y"]/n
    avg_gray= totals["gray"]/n
    avg_people = totals["people"]/n
    avg_faces  = totals["faces"]/n
    avg_motion = totals["motion"]/max(1, n-1)

    # Scene
    if avg_g > 0.18 and avg_b > 0.10:
        scene = "park"
    elif avg_b > 0.25 and avg_y > 0.08:
        scene = "beach"
    elif avg_g > 0.28:
        scene = "field"
    elif avg_gray > 0.55:
        scene = "indoor"
    else:
        scene = "outdoor"

    # Subject
    if avg_faces >= 1.0 or avg_people >= 0.8:
        subject = "person" if (avg_faces < 1.8 and avg_people < 1.6) else "people"
    else:
        subject = "scene"

    # Action from motion + subject
    if avg_motion >= 4.0:
        action = "running" if subject in {"person","people"} else "fast-move"
    elif avg_motion >= 2.2:
        action = "moving" if subject in {"person","people"} else "pan/tilt"
    elif avg_motion >= 1.0:
        action = "walking" if subject in {"person","people"} else "slow-move"
    else:
        action = "talking" if avg_faces >= 1.0 else "static"

    debug = dict(avg_green=avg_g, avg_blue=avg_b, avg_yellow=avg_y, avg_gray=avg_gray,
                 avg_people=avg_people, avg_faces=avg_faces, avg_motion=avg_motion)
    return (scene, subject, action, debug)

# ---------- API ----------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/start_job")
async def start_job(req: Request):
    """
    Client sends: { filename: 'clip.mp4', content_type: 'video/mp4' } (optional)
    Returns: { job_id, upload_url (S3 presigned PUT), object_key }
    """
    try:
        body = await req.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
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

def _process_real(job_id: str, orig_name: str):
    """Download from S3, probe, AI (audio+visual), rename, sidecars, zip, upload."""
    state = JOBS[job_id]

    def step(msg, pct=None):
        state["status"] = msg
        if pct is not None:
            state["pct"] = pct
        state["steps"].append(msg)

    step("Downloading clip...", 5)
    tmpdir = tempfile.mkdtemp(prefix="cliptrial-")
    try:
        input_key = state["object_key"]
        in_name = os.path.basename(input_key)
        local_in = os.path.join(tmpdir, in_name)
        s3.download_file(S3_BUCKET, input_key, local_in)

        step("Reading media info...", 12)
        info = _ffprobe_json(local_in)
        fmt = info.get("format", {}) or {}
        duration = float(fmt.get("duration", 0.0)) if fmt.get("duration") else 0.0

        # Date
        date = None
        tags = (fmt.get("tags") or {})
        ct = tags.get("creation_time")
        if ct:
            try:
                date = datetime.datetime.fromisoformat(ct.replace("Z", "+00:00")).date()
            except Exception:
                pass
        if not date:
            date = datetime.date.today()
        date_str = date.isoformat()

        step("Creating short clip ID...", 18)
        shortid = _sha1_short(local_in)

        # ---------- AUDIO ----------
        has_audio = _has_audio_stream(info)
        if has_audio:
            step("Transcribing audio…", 28)
            try:
                audio_wav = os.path.join(tmpdir, "snippet.wav")
                max_sec = min(WHISPER_MAX_SEC, int(max(15, duration)))
                _extract_audio(local_in, audio_wav, max_seconds=max_sec)
                transcript = _transcribe(audio_wav)
            except Exception:
                transcript = ""
        else:
            step("No audio detected — skipping transcription", 28)
            transcript = ""

        step("Finding subject, action, scene (speech)…", 36)
        scene_t, subject_t, action_t = _labels_from_transcript(transcript)

        # ---------- VISUAL ----------
        step("Analyzing frames…", 44)
        scene_v, subject_v, action_v, dbg = _visual_labels(local_in)

        # Merge: prefer transcript terms when present; otherwise visual
        scene  = scene_t  or scene_v  or "scene"
        subject= subject_t or subject_v or "subject"
        action = action_t or action_v or "action"

        step("Writing searchable metadata...", 55)
        title = f"{subject} {action} at {scene}"
        keywords = [scene, subject, action, f"shortid-{shortid}"]
        _write_metadata(local_in, title=title, keywords=keywords, scene=scene)

        # New name
        step("Building new filename...", 65)
        base, ext = os.path.splitext(in_name)
        ext = ext.lstrip(".").lower() or "mp4"
        new_name = _build_filename(scene, subject, action, date_str, shortid, ext)
        local_out = os.path.join(tmpdir, new_name)
        shutil.copy2(local_in, local_out)

        # Sidecars
        step("Preparing sidecars...", 75)
        resolve_csv = os.path.join(tmpdir, "Resolve.csv")
        _make_resolve_csv(resolve_csv, new_name, scene, "", "", keywords)

        # Zip
        step("Packaging ZIP...", 90)
        zip_path = os.path.join(tmpdir, "result.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(local_out, new_name)
            z.write(resolve_csv, "Resolve.csv")
            z.writestr("manifest.json", json.dumps({
                "original": in_name,
                "renamed": new_name,
                "duration_sec": duration,
                "keywords": keywords,
                "transcript_excerpt": (transcript[:400] + "…") if transcript else "",
                "visual_debug": dbg
            }, indent=2))

        step("Uploading result...", 96)
        with open(zip_path, "rb") as f:
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=state["result_key"],
                Body=f.read(),
                ContentType="application/zip",
            )

        state["filename"] = new_name
        state["status"] = "Done"
        state["pct"] = 100
        state["ready"] = True
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

@app.post("/finalize_upload")
async def finalize_upload(payload: Dict, background: BackgroundTasks):
    job_id = payload.get("job_id")
    orig_name = payload.get("name", "clip.mp4")
    if not job_id or job_id not in JOBS:
        raise HTTPException(status_code=400, detail="Unknown job_id")

    try:
        s3.head_object(Bucket=S3_BUCKET, Key=JOBS[job_id]["object_key"])
    except Exception:
        raise HTTPException(status_code=400, detail="Upload not found in bucket")

    JOBS[job_id]["status"] = "Processing..."
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

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", "Zip not found in S3; fallback path.\n")
    buf.seek(0)
    headers = {
        "Content-Disposition": f"attachment; filename=clip-rename-trial-{job_id}.zip"
    }
    return Response(content=buf.read(), media_type="application/zip", headers=headers)

# ---------- email-gated download ----------

@app.post("/email_link")
async def email_link(payload: Dict):
    email = (payload.get("email") or "").strip()
    job_id = (payload.get("job_id") or "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")
    if job_id not in JOBS or not JOBS[job_id].get("ready"):
        raise HTTPException(status_code=400, detail="Job not ready")

    token = signer.dumps({"job_id": job_id})
    link = f"{APP_BASE_URL}/d/{token}"

    try:
        send_magic_email(email, link)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email send failed: {e}")

    return {"ok": True}

@app.get("/d/{token}")
def direct_download(token: str):
    try:
        data = signer.loads(token, max_age=86400)  # 24h
    except SignatureExpired:
        raise HTTPException(status_code=400, detail="Link expired")
    except BadSignature:
        raise HTTPException(status_code=400, detail="Invalid link")

    job_id = data.get("job_id")
    if not job_id or job_id not in JOBS or not JOBS[job_id].get("ready"):
        raise HTTPException(status_code=404, detail="Job not found or not ready")

    result_key = JOBS[job_id]["result_key"]
    signed = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": result_key},
        ExpiresIn=600,
    )
    return RedirectResponse(url=signed, status_code=302)
