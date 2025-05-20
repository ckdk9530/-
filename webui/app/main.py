from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI(title="CaptureFlow WebUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Sample data – replace with real DB queries in production
# ---------------------------------------------------------------------------
DATA = [
    {"id": 1, "pc": "PC1", "date": "2024-04-01", "path": "sample1.png", "text": "Hello world"},
    {"id": 2, "pc": "PC1", "date": "2024-04-01", "path": "sample2.png", "text": "Another screenshot"},
    {"id": 3, "pc": "PC1", "date": "2024-04-02", "path": "sample3.png", "text": "Keyword example"},
    {"id": 4, "pc": "PC2", "date": "2024-04-01", "path": "sample4.png", "text": "Test image"},
]

@app.get("/api/tree")
def get_tree():
    """Return hierarchical tree PC → Date → Image list."""
    tree = {}
    for row in DATA:
        tree.setdefault(row["pc"], {}).setdefault(row["date"], []).append({
            "id": row["id"], "path": row["path"],
        })
    return tree

@app.get("/api/search")
def search(q: str):
    """Search text from sample data."""
    q_lower = q.lower()
    return [row for row in DATA if q_lower in row["text"].lower()]

@app.get("/api/image/{image_name}")
def get_image(image_name: str):
    fp = STATIC_DIR / image_name
    if not fp.exists():
        raise HTTPException(status_code=404)
    return FileResponse(fp)
