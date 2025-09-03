
# UTA Copilot Backend (FastAPI)

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn apps.api.main:app --reload
```

Open: http://127.0.0.1:8000/docs
