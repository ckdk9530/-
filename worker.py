#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniParser Worker – DB driven, JSON only  (v10 – text‑only filter)
================================================================
* **Debug mode**  (`--debug [--img path]`)
    • 單次推論指定影像 → 只保留 `type == "text"` 的項目並 `print(JSON)`。
* **Normal mode**
    • 多執行緒輪詢 DB，處理 `pending`。
    • 推論結果同樣過濾後寫回 `json_payload`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import torch
from PIL import Image

# ───────────────────────────
# CLI – early parse
# ───────────────────────────
cli_parser = argparse.ArgumentParser()
cli_parser.add_argument("--debug", action="store_true", help="run single‑image debug mode")
cli_parser.add_argument("--img", help="image path for debug mode")
args, _ = cli_parser.parse_known_args()
DEBUG = args.debug
DEBUG_IMG = args.img

# ───────────────────────────
# Model (inline omni_parse_json) – returns TEXT items only
# ───────────────────────────
from OmniParser.util.utils import (
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
    get_som_labeled_img,
    _get_paddle_ocr,            # 單例取得 PaddleOCR
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = get_yolo_model(model_path="weights/icon_detect/model.pt")
try:
    yolo_model = yolo_model.to(DEVICE)  # ignore if .to not available
except AttributeError:
    pass

caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence",
)

# 建立全域 PaddleOCR 單例，避免重複 build predictor
GLOBAL_PADDLE_OCR = _get_paddle_ocr(use_gpu=(DEVICE.type == "cuda"))

@torch.inference_mode()
def omni_parse_json(
    image_path: str | Path,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.10,
    use_paddleocr: bool = True,
    imgsz: int = 640,
):
    img = Image.open(image_path).convert("RGB")
    (ocr_text, ocr_bbox), _ = check_ocr_box(
        img,
        display_img=False,
        output_bb_format="xyxy",
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=use_paddleocr,
        paddle_ocr=GLOBAL_PADDLE_OCR,
    )
    _, _, parsed = get_som_labeled_img(
        img,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        caption_model_processor=caption_model_processor,
        ocr_text=ocr_text,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
    # ─── Filter: keep only items where type == "text" ───
    parsed = [item.get("content", "") for item in parsed if item.get("content")]
    return parsed

# ───────────────────────────
# Debug – run & exit early
# ───────────────────────────
if DEBUG:
    img_path = Path(DEBUG_IMG) if DEBUG_IMG else Path(input("Image path » ").strip())
    parsed = omni_parse_json(img_path)
    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    sys.exit(0)

# ───────────────────────────
# Normal‑mode Config & DB deps (lazy import)
# ───────────────────────────
LOG_LEVEL   = "INFO"
THREADS     = 4
BATCH_SIZE  = 10

DB_URL = (
    "postgresql+psycopg2://omniapp:"  # noqa: E501
    "uG6#9jT!wZ8rL@p2$Xv4qS1e%Nb0Ka7C"  # pragma: allowlist secret
    "@127.0.0.1:5432/screencap"
)

DB_PREFIX    = Path("/volume1/ScreenshotService")
LOCAL_PREFIX = Path("/mnt/nas/inbox")

from sqlalchemy import create_engine, text  # noqa: E402

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
engine = create_engine(DB_URL, pool_size=THREADS*2, max_overflow=0)

# ───────────────────────────
# Helper functions
# ───────────────────────────

def db_to_local(p):
    p = Path(p)
    try:
        rel = p.relative_to(DB_PREFIX)
        return LOCAL_PREFIX / rel
    except ValueError:
        return p

def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ───────────────────────────
# worker_stats helpers
# ───────────────────────────
MAC_ADDR = "omni-worker"

def stats_init(conn):
    conn.execute(text("""
        INSERT INTO worker_stats (mac_address, processed_ok, processed_err, pending, last_update, current_img)
        VALUES (:m, 0, 0, 0, now(), NULL)
        ON CONFLICT (mac_address) DO NOTHING;
    """), dict(m=MAC_ADDR))

def stats_claim(conn, n):
    conn.execute(text("UPDATE worker_stats SET pending = pending + :n, last_update = now() WHERE mac_address = :m"), dict(n=n, m=MAC_ADDR))

def stats_progress(conn, cur):
    conn.execute(text("UPDATE worker_stats SET current_img = :c, last_update = now() WHERE mac_address = :m"), dict(c=cur, m=MAC_ADDR))

def stats_done(conn, ok=0, err=0):
    conn.execute(text("""
        UPDATE worker_stats
           SET processed_ok  = processed_ok  + :ok,
               processed_err = processed_err + :er,
               pending       = pending - (:ok + :er),
               last_update   = now(),
               current_img   = NULL
         WHERE mac_address = :m;
    """), dict(ok=ok, er=err, m=MAC_ADDR))

# ───────────────────────────
# Core DB ops
# ───────────────────────────

def claim_tasks(limit):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            UPDATE captures
               SET status = 'processing'
             WHERE id IN (
                   SELECT id FROM captures WHERE status = 'pending' LIMIT :lim FOR UPDATE SKIP LOCKED
             )
            RETURNING id, img_path, sha256_img;
        """), dict(lim=limit)).mappings().all()
        if rows:
            stats_claim(conn, len(rows))
        return rows

def update_capture_done(conn, cid, js, sha):
    conn.execute(text("""
        UPDATE captures
           SET json_payload = :j,
               sha256_img   = :s,
               status       = 'done'
         WHERE id = :cid;
    """), dict(j=js, s=sha, cid=cid))

def mark_error(conn, cid):
    conn.execute(text("UPDATE captures SET status = 'err' WHERE id = :cid"), dict(cid=cid))

# ───────────────────────────
# Worker thread
# ───────────────────────────

def handle_row(row):
    cid = row["id"]
    db_path = row["img_path"]
    sha_db  = row["sha256_img"] or ""
    local   = db_to_local(db_path)

    if not local.exists():
        raise FileNotFoundError(local)

    sha_now = sha256_file(local)
    if sha_db and sha_db != sha_now:
        raise ValueError("sha256_img mismatch")

    parsed = omni_parse_json(local)
    return sha_now, parsed


def worker_loop():
    while True:
        tasks = claim_tasks(BATCH_SIZE)
        if not tasks:
            time.sleep(2)
            continue
        for row in tasks:
            try:
                with engine.begin() as conn:
                    stats_progress(conn, row["img_path"])
                sha_now, parsed = handle_row(row)
                with engine.begin() as conn:
                    update_capture_done(conn, row["id"], json.dumps(parsed, ensure_ascii=False), sha_now)
                    stats_done(conn, ok=1)
                logging.info("DONE id=%d", row["id"])
            except Exception:
                logging.exception("ERR  id=%d", row["id"])
                with engine.begin() as conn:
                    mark_error(conn, row["id"])
                    stats_done(conn, err=1)

# ───────────────────────────
# Boot normal mode
# ───────────────────────────
if __name__ == "__main__":
    with engine.begin() as conn:
        stats_init(conn)
    logging.info("Worker started (threads=%d)", THREADS)
    for _ in range(THREADS):
        threading.Thread(target=worker_loop, daemon=True).start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Shutdown – bye")
