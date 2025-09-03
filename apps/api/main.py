
import os
import json
import datetime as dt
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import ics
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import uuid, pathlib
from openai import OpenAI


# Basic env defaults (you can also use a .env loader if you like)
UTA_EVENTS_ICS = os.getenv("UTA_EVENTS_ICS", "https://events.uta.edu/calendar.ics")
UTA_DINING_BASE = os.getenv("UTA_DINING_BASE", "https://dineoncampus.com/utarlington")
UTA_AVG_COST = os.getenv("UTA_AVG_COST", "https://www.uta.edu/administration/fao/average-cost")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    user_profile: Optional[dict] = None

app = FastAPI(title="UTA Copilot API")

# Allow mobile app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create folders for storing temporary files and calendar events
DATA_DIR = pathlib.Path("data"); DATA_DIR.mkdir(exist_ok=True)
ICS_DIR = DATA_DIR / "ics"; ICS_DIR.mkdir(exist_ok=True)


async def fetch_events(ical_url: Optional[str] = None, limit: int = 10):
    src = ical_url or UTA_EVENTS_ICS
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(src)
        r.raise_for_status()
    cal = ics.Calendar(r.text)
    upcoming = []
    now = dt.datetime.now(dt.timezone.utc)
    # Some feeds may be unsorted; we sort by start time
    for ev in sorted(cal.events, key=lambda e: e.begin or now):
        try:
            begin_dt = ev.begin.datetime if ev.begin else None
            if begin_dt and begin_dt >= now:
                upcoming.append({
                    "title": ev.name,
                    "begin": ev.begin.format("YYYY-MM-DD HH:mm"),
                    "location": ev.location,
                })
                if len(upcoming) >= limit:
                    break
        except Exception:
            continue
    return {"events": upcoming}

async def fetch_dining_today(base: str = None):
    # Stub; returns proof-of-life and a couple of common venues.
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
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(target)
        r.raise_for_status()
    # We return length only (no scraping) for now
    return {"source": target, "html_length": len(r.text)}

@app.get("/health")
async def health():
    return {"ok": True, "ts": dt.datetime.utcnow().isoformat()}

@app.post("/chat")
async def chat(req: ChatRequest):
    last = (req.messages[-1].content if req.messages else "").lower()
    # Very simple router for the starter
    if "event" in last or "happening" in last:
        data = await fetch_events()
        return {"name": "events", "content": json.dumps(data)}
    if "dining" in last or "menu" in last:
        data = await fetch_dining_today()
        return {"name": "dining", "content": json.dumps(data)}
    if "tuition" in last or "average cost" in last or "cost" in last:
        data = await fetch_average_cost()
        return {"name": "avg_cost", "content": json.dumps(data)}
    # Default
    return {"name": "answer", "content": "RAG answer placeholder. Try asking about events, dining, or cost."}

# --- Speech to Text (Whisper) ---
@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse({"error":"OPENAI_API_KEY not set"}, status_code=500)

    tmp_path = DATA_DIR / f"upload_{uuid.uuid4().hex}_{file.filename}"
    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        client = OpenAI(api_key=api_key)
        with tmp_path.open("rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        text = transcript.text if hasattr(transcript, "text") else str(transcript)
        return {"text": text}
    finally:
        try: tmp_path.unlink()
        except: pass

# --- Calendar (ICS) ---
def build_ics(title: str, begin_iso: str, end_iso: str, location: str = None) -> pathlib.Path:
    import ics
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

@app.post("/calendar/create")
async def calendar_create(payload: dict):
    try:
        title = payload.get("title")
        begin = payload.get("begin")
        end = payload.get("end")
        location = payload.get("location")
        ics_path = build_ics(title, begin, end, location)
        return {
            "url": f"/calendar/{ics_path.name}",
            "filename": ics_path.name
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/calendar/{fname}")
async def calendar_download(fname: str):
    f = ICS_DIR / fname
    if not f.exists():
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(path=f, media_type="text/calendar", filename=fname)
