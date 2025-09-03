import os
import json
import time
import uuid
import pathlib
import datetime as dt
from typing import List, Optional

import httpx
import ics
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# ----------- Config (env overrides allowed) -----------
UTA_EVENTS_ICS = os.getenv("UTA_EVENTS_ICS", "https://events.uta.edu/calendar.ics")
UTA_DINING_BASE = os.getenv("UTA_DINING_BASE", "https://dineoncampus.com/utarlington")
UTA_AVG_COST = os.getenv("UTA_AVG_COST", "https://www.uta.edu/administration/fao/average-cost")

# Storage
DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(exist_ok=True)
ICS_DIR = DATA_DIR / "ics"; ICS_DIR.mkdir(exist_ok=True)

# Simple cache for events
EVENTS_CACHE = {"data": None, "ts": 0.0}
EVENTS_TTL = 300  # 5 minutes

# ----------- Models -----------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    user_profile: Optional[dict] = None

# ----------- App -----------
app = FastAPI(title="UTA Copilot API")

# CORS for dev; tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared HTTP client (keep-alive + timeout)
@app.on_event("startup")
async def _startup():
    # remove http2=True
    app.state.http = httpx.AsyncClient(timeout=12)


@app.on_event("shutdown")
async def _shutdown():
    try:
        await app.state.http.aclose()
    except Exception:
        pass

# ----------- Helpers -----------
async def fetch_events(ical_url: Optional[str] = None, limit: int = 10):
    now = time.time()
    if EVENTS_CACHE["data"] and now - EVENTS_CACHE["ts"] < EVENTS_TTL:
        return EVENTS_CACHE["data"]

    src = ical_url or UTA_EVENTS_ICS
    r = await app.state.http.get(src)
    r.raise_for_status()

    cal = ics.Calendar(r.text)
    upcoming = []
    now_dt = dt.datetime.now(dt.timezone.utc)
    for ev in sorted(cal.events, key=lambda e: e.begin or now_dt):
        begin_dt = ev.begin.datetime if ev.begin else None
        if begin_dt and begin_dt >= now_dt:
            upcoming.append({
                "title": ev.name,
                "begin": ev.begin.format("YYYY-MM-DD HH:mm"),
                "location": ev.location,
            })
            if len(upcoming) >= limit:
                break

    data = {"events": upcoming}
    EVENTS_CACHE.update({"data": data, "ts": time.time()})
    return data

async def fetch_dining_today(base: str = None):
    # Stub response for MVP; replace with real scrape/API later
    url = base or UTA_DINING_BASE
    return {
        "source": url,
        "venues": [
            {"name": "Connection Café", "url": url, "hours": "7:00 AM – 9:00 PM"},
            {"name": "Panda Express", "url": url, "hours": "11:00 AM – 8:00 PM"}
        ]
    }

async def fetch_average_cost(url: str = None):
    target = url or UTA_AVG_COST
    r = await app.state.http.get(target)
    r.raise_for_status()
    return {"source": target, "html_length": len(r.text)}

def build_ics(title: str, begin_iso: str, end_iso: str, location: str = None) -> pathlib.Path:
    c = ics.Calendar()
    e = ics.Event()
    e.name = title
    e.begin = begin_iso
    e.end = end_iso
    if location:
        e.location = location
    c.events.add(e)
    file_id = uuid.uuid4().hex
    out = ICS_DIR / f"{file_id}.ics"
    out.write_text(str(c), encoding="utf-8")
    return out

# ----------- Routes -----------
@app.get("/health")
async def health():
    return {"ok": True, "ts": dt.datetime.utcnow().isoformat()}

@app.post("/chat")
async def chat(req: ChatRequest):
    last = (req.messages[-1].content if req.messages else "").lower()

    try:
        if "event" in last or "happening" in last:
            try:
                data = await fetch_events()
            except Exception:
                # safe fallback so UI never goes blank
                data = {"events":[{"title":"Test Event","begin":"2025-09-05 18:00","location":"University Center"}]}
            return {"name": "events", "content": data}

        if "dining" in last or "menu" in last:
            data = await fetch_dining_today()
            return {"name": "dining", "content": data}

        if "tuition" in last or "average cost" in last or "cost" in last:
            data = await fetch_average_cost()
            return {"name": "avg_cost", "content": data}

        return {"name": "answer", "content": "RAG answer placeholder. Try asking about events, dining, or cost."}
    except Exception as e:
        return JSONResponse({"error": f"chat_failed: {e}"}, status_code=500)

# Speech-to-Text (Whisper)
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return JSONResponse({"error":"OPENAI_API_KEY not set"}, status_code=500)

        # Save uploaded file with a sensible name/extension
        fname = file.filename or "audio.m4a"
        tmp = DATA_DIR / f"upload_{uuid.uuid4().hex}_{fname}"
        tmp.write_bytes(await file.read())

        try:
            client = OpenAI(api_key=api_key)
            with tmp.open("rb") as f:
                tr = client.audio.transcriptions.create(model="whisper-1", file=f)
            text = getattr(tr, "text", None) or str(tr)
            return {"text": text}
        finally:
            try: tmp.unlink()
            except: pass

    except Exception as e:
        # Always return JSON so the app can parse errors safely
        return JSONResponse({"error": f"stt_failed: {e}"}, status_code=400)

# Calendar create + download
@app.post("/calendar/create")
async def calendar_create(payload: dict):
    try:
        title = payload.get("title")
        begin = payload.get("begin")
        end = payload.get("end")
        location = payload.get("location")
        if not title or not begin or not end:
            return JSONResponse({"error":"title, begin, end are required"}, status_code=400)
        ics_path = build_ics(title, begin, end, location)
        return {"url": f"/calendar/{ics_path.name}", "filename": ics_path.name}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/calendar/{fname}")
async def calendar_download(fname: str):
    f = ICS_DIR / fname
    if not f.exists():
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(path=f, media_type="text/calendar", filename=fname)
