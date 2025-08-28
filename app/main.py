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
from typing import Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response, JSONResponse

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
    return "".join(ch.lower() if c
