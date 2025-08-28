# app/main.py
import os
import io
import re
import json
import time
import uuid
import shutil
import boto3
import zipfile
import hashlib
import tempfile
import mimetypes
import subprocess
import datetime
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
FROM_EMAIL     = os.environ.get("FROM_EMAIL", "no-reply@example.com")  # must be a verified sender in Postmark

WHISPER_MODEL  = os.environ.get("WHISPER_MODEL", "tiny")   # tiny/base for CPU
WHISPER_MAX_SEC= int(os.environ.get("WHISPER_MAX_SEC", "120"))  # cap transcript to 2 min audio
CPU_THREADS    = int(os.environ.get("CPU_THREADS", "4"))

s3 = boto3.client("s3", region_name=AWS_REGION)
signer = URLSafeTimedSerializer(SECRET_KEY)

# lazy-loaded models
_nlp = None
_whisper = None

app = FastAPI(title="Clip Rename Trial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Framer domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (OK for the trial)
JOBS: Dict[str, Dict] = {}

# ---------- small utils ----------

def _run(cmd: list, check=True) -> str:
    """Run a subprocess and return stdout text."""
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

def _sha1_short(path: str, bytes_limit: int = 8_000_000) -> str:
    """Fast file ID: SHA1 of the first N bytes."""
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
    """No-op for now (ExifTool not installed in this build)."""
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
            "From": FROM_EMAIL,   # must be verified in Postmark
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
        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])  # fast
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
    # lightweight decode settings for CPU
    segments, info = model.transcribe(
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
        if sum(len(x) for x in out) > 4000:  # cap text length
            break
    return " ".join(out).strip()

def _most_common(items: List[str]) -> str:
    items = [x for x in items if x]
    return Counter(items).most_common(1)[0][0] if items else ""

def _labels_from_transcript(text: str) -> Tuple[str, str, str]:
    """
    Returns (scene, subject, action) from transcript text.
    - subject: most common PERSON entity; fallback to top Proper Noun/Noun chunk
    - action: most common verb lemma (non-aux), short
    - scene: look for 'at|in|on|near the|a ...' noun phrase; else LOC/GPE entity; else top noun chunk
    """
    if not text or len(text) < 5:
        return ("scene", "subject", "action")

    nlp = _load_nlp()
    doc = nlp(text)

    # SUBJECT (PERSON)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    subject = _most_common(persons)
    if not subject:
        # fallback: proper nouns
        proper = [t.text for t in doc if t.pos_ == "PROPN" and len(t.text) > 1]
        subject = _most_common(proper)
    if not subject:
        # fallback: noun chunks
        noun_chunks = [nc.root.text for nc in doc.noun_chunks]
        subject = _most_common(noun_chunks) or "subject"

    # ACTION (VERB)
    verbs = [t.lemma_ for t in doc if t.pos_ == "VERB" and t.lemma_ not in {"be","do","have"}]
    action = _most_common(verbs) or "action"

    # SCENE (prepositional phrase → places)
    scene = ""
    # simple regex for " at/in/o
