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
from typing import Dict, List, Tuple, Callable, Optional
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
CPU_THREADS     = int(os.environ.get("CPU_THREADS", "2"))

# ---- Visual analysis (stride across the WHOLE clip; no time budget by default) ----
VIS_FRAME_STRIDE        = int(os.environ.get("VIS_FRAME_STRIDE", "8"))   # analyze 1 of every N frames
VIS_MAX_ANALYSIS_FRAMES = int(os.environ.get("VIS_MAX_ANALYSIS_FRAMES", "0"))  # 0 = unlimited
VIS_TIME_BUDGET_SEC     = float(os.environ.get("VIS_TIME_BUDGET_SEC", "0"))    # 0 = no time cap
VIS_USE_HOG             = os.environ.get("VIS_USE_HOG", "1") == "1"      # people detector (slower)
VIS_MAX_EDGE            = int(os.environ.get("VIS_MAX_EDGE", "480"))     # resize so max(h,w)=this

# Filename templating (underscores between chips, hyphens inside chips)
FILENAME_TEMPLATE = os.environ.get(
    "FILENAME_TEMPLATE",
    "{scene}_{subject}_{action}_{date}_{orig}"
)
MAX_FILENAME_BYTES = int(os.environ.get("MAX_FILENAME_BYTES", "96"))
ORIG_TRUNC         = int(os.environ.get("ORIG_TRUNC", "24"))

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

def _parse_fps(ffprobe_info: dict) -> float:
    # Try video stream avg_frame_rate; fallback to r_frame_rate
    streams = ffprobe_info.get("streams") or []
    for s in streams:
        if s.get("codec_type") == "video":
            rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "0/1"
            try:
                num, den = rate.split("/")
                num = float(num); den = float(den) if float(den) != 0 else 1.0
                fps = num / den
                if fps > 1.0:
                    return fps
            except Exception:
                pass
    # fallback generic
    return 30.0

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
    """Lower, keep alnum, hyphen as word separator (inside chips)."""
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in (s or "")).strip("-").replace("--", "-")

def _truncate_token(s: str, maxlen: int) -> str:
    s = s[:maxlen]
    return re.sub(r"-{2,}", "-", s).strip("-")

def _compose_filename_from_template(template: str, tokens: Dict[str, str], ext: str) -> str:
    """
    Build filename using underscores between chips and hyphens inside chips.
    Empty chips are skipped. Enforce byte cap by shrinking in priority order.
    """
    raw_parts = [p for p in template.split("_") if p]
    chips: List[Tuple[str,str]] = []
    for part in raw_parts:
        m = re.fullmatch(r"\{([a-z0-9_]+)\}", part.strip())
        if not m:
            continue
        key = m.group(1)
        val = (tokens.get(key) or "").strip()
        if not val:
            continue
        val_slug = _safe_slug(val)
        if key == "orig":
            val_slug = _truncate_token(val_slug, ORIG_TRUNC)
        chips.append((key, val_slug))

    base = "_".join(v for _, v in chips if v)
    reserve = 1 + len(ext)
    max_bytes = max(16, MAX_FILENAME_BYTES - reserve)

    if len(base.encode("utf-8")) > max_bytes:
        priority = ["scene", "action", "subject", "orig"]
        name_to_idx = {name: i for i, (name, _) in enumerate(chips)}
        limits = [len(val) for _, val in chips]

        def shrink_one(target_key: str, dec: int = 4):
            if target_key not in name_to_idx:
                return False
            i = name_to_idx[target_key]
            if limits[i] <= 8:
                return False
            limits[i] = max(8, limits[i] - dec)
            return True

        guard = 0
        while len("_".join(_truncate_token(val, limits[i]) for i, (_, val) in enumerate(chips)).encode("utf-8")) > max_bytes and guard < 100:
            progressed = False
            for key in priority:
                progressed = shrink_one(key) or progressed
                if len("_".join(_truncate_token(val, limits[i]) for i, (_, val) in enumerate(chips)).encode("utf-8")) <= max_bytes:
                    break
            if not progressed:
                for i in range(len(limits)):
                    limits[i] = max(6, int(limits[i] * 0.9))
            guard += 1

        base = "_".join(_truncate_token(val, limits[i]) for i, (_, val) in enumerate(chips))

    return base + f".{ext}"

def _ensure_unique_path(dirpath: str, filename: str) -> str:
    """Append _02, _03, ... if needed to avoid collision in dirpath."""
    stem, ext = os.path.splitext(filename)
    cand = filename
    n = 2
    while os.path.exists(os.path.join(dirpath, cand)):
        cand = f"{stem}_{n:02d}{ext}"
        n += 1
        if n > 99:
            break
    return cand

def _write_metadata(path: str, title: str, keywords: list, scene: str):
    # No-op in this build (ExifTool not installed here).
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
        timeout=30,
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
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        except Exception:
            import spacy
            _nlp = spacy.blank("en")
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
        proper = [t.text for t in doc if getattr(t, "pos_", "") == "PROPN" and len(t.text) > 1]
        subject = _most_common(proper)
    if not subject:
        noun_chunks = []
        try:
            noun_chunks = [nc.root.text for nc in doc.noun_chunks]
        except Exception:
            pass
        subject = _most_common(noun_chunks)

    verbs = [getattr(t, "lemma_", t.text) for t in doc if getattr(t, "pos_", "") == "VERB" and getattr(t, "lemma_", t.text) not in {"be","do","have"}]
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

# ---------- Visual cues (CPU, OpenCV) — stride sampling across the FULL clip ----------

_PEOPLE_HOG = None
_FACE_CASCADE = None

def _load_detectors():
    global _PEOPLE_HOG, _FACE_CASCADE
    if VIS_USE_HOG and _PEOPLE_HOG is None:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _PEOPLE_HOG = hog
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _detect_people_and_faces(bgr_img: np.ndarray) -> Tuple[int, int]:
    _load_detectors()
    # faces (cheap)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48))
    people = 0
    if VIS_USE_HOG:
        h, w = bgr_img.shape[:2]
        target_w = 480
        if w > target_w:
            img_small = cv2.resize(bgr_img, (target_w, int(h * target_w / max(1, w))))
        else:
            img_small = bgr_img
        rects, _ = _PEOPLE_HOG.detectMultiScale(img_small, winStride=(8,8), padding=(8,8), scale=1.05)
        people = len(rects)
    return people, len(faces)

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

def _motion_magnitude(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=2, winsize=15,
                                        iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=False)
    return float(np.mean(mag))

def _visual_labels_stride(video_path: str, ffinfo: dict, progress_cb: Optional[Callable[[str], None]] = None) -> Tuple[str, str, str, dict]:
    """
    Analyze EVERY Nth frame (VIS_FRAME_STRIDE) across the whole clip.
    No time cap by default (VIS_TIME_BUDGET_SEC = 0). Processes frames streaming (constant memory).
    """
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ("scene", "subject", "action", {"frames": 0, "note": "open failed"})

    total_frames_cap = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_cap = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fps_probe = _parse_fps(ffinfo)
    dur = 0.0
    try:
        dur = float((ffinfo.get("format") or {}).get("duration") or 0.0)
    except Exception:
        pass

    # Estimate total frames if CAP reports 0
    if total_frames_cap <= 0 and dur > 0 and fps_probe > 1.0:
        total_frames_est = int(dur * fps_probe)
    else:
        total_frames_est = total_frames_cap if total_frames_cap > 0 else int(dur * (fps_cap if fps_cap > 1 else fps_probe))

    step = max(1, int(VIS_FRAME_STRIDE))
    indices = list(range(0, max(0, total_frames_est), step))
    if VIS_MAX_ANALYSIS_FRAMES > 0:
        indices = indices[:VIS_MAX_ANALYSIS_FRAMES]

    totals = dict(g=0.0,b=0.0,y=0.0,gray=0.0, people=0, faces=0, motion=0.0, frames=0)
    prev_gray = None
    analyzed = 0
    total_targets = len(indices)

    for i, idx in enumerate(indices, 1):
        # Optional wall-clock guard if user sets a budget
        if VIS_TIME_BUDGET_SEC > 0 and (time.time() - t0) > VIS_TIME_BUDGET_SEC:
            break

        # Seek and read
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # resize to VIS_MAX_EDGE
        h, w = frame.shape[:2]
        m = max(h, w)
        if m > VIS_MAX_EDGE:
            scale = VIS_MAX_EDGE / float(m)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

        # metrics
        g, b, y, gr = _color_ratios(frame)
        p, fa = _detect_people_and_faces(frame)
        totals["g"] += g; totals["b"] += b; totals["y"] += y; totals["gray"] += gr
        totals["people"] += p; totals["faces"] += fa

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            totals["motion"] += _motion_magnitude(prev_gray, gray)
        prev_gray = gray

        analyzed += 1
        totals["frames"] = analyzed

        if progress_cb and (i % 25 == 0 or i == total_targets):
            progress_cb(f"Analyzing frames ({i}/{total_targets})…")

    cap.release()

    if analyzed == 0:
        return ("scene", "subject", "action", {"frames": 0, "note": "no frames decoded"})

    n = analyzed
    avg_g   = totals["g"]/n
    avg_b   = totals["b"]/n
    avg_y   = totals["y"]/n
    avg_gray= totals["gray"]/n
    avg_people = totals["people"]/n
    avg_faces  = totals["faces"]/n
    avg_motion = totals["motion"]/max(1, n-1)

    # Scene heuristic
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

    # Action
    if avg_motion >= 4.0:
        action = "running" if subject in {"person","people"} else "fast-move"
    elif avg_motion >= 2.2:
        action = "moving" if subject in {"person","people"} else "pan-tilt"
    elif avg_motion >= 1.0:
        action = "walking" if subject in {"person","people"} else "slow-move"
    else:
        action = "talking" if avg_faces >= 1.0 else "static"

    debug = dict(
        avg_green=avg_g, avg_blue=avg_b, avg_yellow=avg_y, avg_gray=avg_gray,
        avg_people=avg_people, avg_faces=avg_faces, avg_motion=avg_motion,
        frames_analyzed=n, hog_used=VIS_USE_HOG, time_spent_sec=round(time.time()-t0,2),
        stride=step, estimated_total_frames=total_frames_est
    )
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

        # Date & Time
        date = None
        shoot_dt = None
        tags = (fmt.get("tags") or {})
        ct = tags.get("creation_time")
        if ct:
            try:
                shoot_dt = datetime.datetime.fromisoformat(ct.replace("Z", "+00:00"))
                date = shoot_dt.date()
            except Exception:
                pass
        if not date:
            shoot_dt = datetime.datetime.utcnow()
            date = shoot_dt.date()
        date_str = date.isoformat()
        time_str = shoot_dt.strftime("%H%M%S")

        step("Creating clip UID…", 18)
        uid = _sha1_short(local_in)  # hidden, not used in filename

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

        # ---------- VISUAL (stride across whole clip) ----------
        def cb(msg: str):
            step(msg)

        step("Analyzing frames…", 44)
        scene_v, subject_v, action_v, dbg = _visual_labels_stride(local_in, info, progress_cb=cb)

        # Merge: prefer transcript terms when present; otherwise visual
        scene   = scene_t   or scene_v  or "scene"
        subject = subject_t or subject_v or "subject"
        action  = action_t  or action_v or "action"

        # ---------- Filename templating ----------
        base, ext = os.path.splitext(in_name)
        ext = ext.lstrip(".").lower() or "mp4"

        tokens = {
            "scene": scene,
            "subject": subject,
            "action": action,
            "date": date_str,
            "time": time_str,
            "orig": base,              # original filename stem (user-friendly)
            "camera": "",              # reserved
            "card": "",                # reserved
            "seq": "",                 # reserved
            "uid": uid,                # hidden
        }
        new_name = _compose_filename_from_template(FILENAME_TEMPLATE, tokens, ext)
        new_name = _ensure_unique_path(tmpdir, new_name)

        step("Writing searchable metadata...", 55)
        title = f"{subject} {action} at {scene}"
        keywords = [
            scene, subject, action,
            f"uid-{uid}",                 # hidden but Spotlight-searchable
            f"orig-{_safe_slug(base)[:ORIG_TRUNC]}"
        ]
        _write_metadata(local_in, title=title, keywords=keywords, scene=scene)

        # Copy to final name
        step("Building new filename...", 65)
        local_out = os.path.join(tmpdir, new_name)
        shutil.copy2(local_in, local_out)

        # Sidecars
        step("Preparing sidecars...", 75)
        resolve_csv = os.path.join(tmpdir, "Resolve.csv")
        _make_resolve_csv(resolve_csv, new_name, scene, "", "", keywords)

        # Manifest + ZIP
        step("Packaging ZIP...", 90)
        zip_path = os.path.join(tmpdir, "result.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(local_out, new_name)
            z.write(resolve_csv, "Resolve.csv")
            z.writestr("manifest.json", json.dumps({
                "original": in_name,
                "renamed": new_name,
                "duration_sec": duration,
                "tokens": tokens,
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
