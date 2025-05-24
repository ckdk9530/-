#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniParser Worker – DB driven, JSON only  (v16 – PaddleOCR singleton)
====================================================================
* v12: 修正 get_yolo_model 無 device 參數
* v13: 移除 get_som_labeled_img 的 yolo_result
* v14: 在 --debug-dir 模式新增統計 (count / total / avg / min / max)
* v15: --debug-dir 執行時改為輸出進度，不列印 JSON 內容
* v16: 建立 PaddleOCR 單例，顯示傳入 check_ocr_box()，徹底避免多次 Predictor
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

import contextlib
import io
import gc

import paddle
import torch
from PIL import Image

# ───────────────────────────
# CLI
# ───────────────────────────
cli = argparse.ArgumentParser()
cli.add_argument("--debug", action="store_true", help="單張 debug")
cli.add_argument("--img", help="單張路徑")
cli.add_argument("--debug-dir", help="資料夾批量 debug")
cli.add_argument("--batch-size", type=int, default=8, help="一次 GPU 推論張數")
args, _ = cli.parse_known_args()
DEBUG = args.debug or bool(args.debug_dir)
DEBUG_IMG: str | None = args.img
DEBUG_DIR: str | None = args.debug_dir
BATCH_SIZE = args.batch_size

# ───────────────────────────
# Model utils
# ───────────────────────────
from OmniParser.util.utils import (
    check_ocr_box,
    get_caption_model_processor,
    get_som_labeled_img,
    get_yolo_model,
    _get_paddle_ocr,                # <──── util 中的單例 helper
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = get_yolo_model("weights/icon_detect/model.pt").to(DEVICE)
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence",
    device=DEVICE,
)

# ==== 變更點 1：在主程式先建好 PaddleOCR 單例 ====
GLOBAL_PADDLE_OCR = _get_paddle_ocr(use_gpu=(DEVICE == "cuda"))
# ================================================

# ───────────────────────────
# Inference helpers
# ───────────────────────────
@torch.inference_mode()
def omni_parse_json_single(
    image_path: Path,
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
        paddle_ocr=GLOBAL_PADDLE_OCR,          # <── 顯式傳入
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
    return [i.get("content", "") for i in parsed if i.get("content")]


@torch.inference_mode()
def omni_parse_json_batch(
    image_paths: List[Path],
    box_threshold: float = 0.05,
    iou_threshold: float = 0.10,
    use_paddleocr: bool = True,
    imgsz: int = 640,
) -> List[List[str]]:
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    _ = yolo_model.predict(imgs, batch=len(imgs), verbose=False)

    outputs: List[List[str]] = []
    for img in imgs:
        (ocr_text, ocr_bbox), _ = check_ocr_box(
            img,
            display_img=False,
            output_bb_format="xyxy",
            easyocr_args={"paragraph": False, "text_threshold": 0.9},
            use_paddleocr=use_paddleocr,
            paddle_ocr=GLOBAL_PADDLE_OCR,      # <── 顯式傳入
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
        outputs.append([i.get("content", "") for i in parsed if i.get("content")])

    # 主動清理未再使用的張量並釋放 GPU 快取
    del imgs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        paddle.device.cuda.empty_cache()

    return outputs

# ───────────────────────────
# Debug shortcuts
# ───────────────────────────
if DEBUG and not DEBUG_DIR:
    p = Path(DEBUG_IMG) if DEBUG_IMG else Path(input("Image path » ").strip())
    print(json.dumps(omni_parse_json_single(p), ensure_ascii=False, indent=2))
    sys.exit()

if DEBUG_DIR:
    imgs = sorted(
        [p for p in Path(DEBUG_DIR).rglob("*")
         if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )
    print(f"Total images: {len(imgs)}")
    start_all = time.perf_counter()
    durations: List[float] = []
    processed = 0

    def _hms(sec: float) -> str:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02}:{m:02}:{s:02}"

    for i in range(0, len(imgs), BATCH_SIZE):
        batch = imgs[i:i + BATCH_SIZE]
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            _ = omni_parse_json_batch(batch)
        t1 = time.perf_counter()

        per_img = (t1 - t0) / len(batch)
        durations.extend([per_img] * len(batch))

        processed += len(batch)
        elapsed = time.perf_counter() - start_all
        avg_per_img = sum(durations) / len(durations)
        remaining = max(len(imgs) - processed, 0) * avg_per_img

        print(
            f"{processed}/{len(imgs)} {per_img:.3f}s "
            f"elapsed {_hms(elapsed)} remain {_hms(remaining)}"
        )

    total_time = time.perf_counter() - start_all
    print("\n=== Summary ===")
    print(f"Images : {len(imgs)}")
    print(f"Total  : {total_time:.3f}s")
    if imgs:
        print(f"Avg    : {total_time / len(imgs):.3f}s")
        print(f"Min    : {min(durations):.3f}s")
        print(f"Max    : {max(durations):.3f}s")
    sys.exit()

# ───────────────────────────
# Normal-mode Config & DB deps
# ───────────────────────────
LOG_LEVEL = "INFO"
THREADS = 4           # DB 輪詢 / commit 執行緒數
BATCH_SIZE = 8        # ➜ GPU 一次推論張數 / DB claim

DB_URL = (
    "postgresql+psycopg2://omniapp:"
    "uG6#9jT!wZ8rL@p2$Xv4qS1e%Nb0Ka7C"
    "@127.0.0.1:5432/screencap"
)

DB_PREFIX = Path("/volume1/ScreenshotService")
LOCAL_PREFIX = Path("/mnt/nas/inbox")

from sqlalchemy import create_engine, text  # noqa: E402

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
engine = create_engine(DB_URL, pool_size=THREADS * 2, max_overflow=0)

# ───────────────────────────
# Helper functions (unchanged)
# ───────────────────────────

def db_to_local(p: str | Path) -> Path:
    p = Path(p)
    try:
        return LOCAL_PREFIX / p.relative_to(DB_PREFIX)
    except ValueError:
        return p

def sha256_file(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

# ───────────────────────────
# worker_stats helpers (unchanged)
# ───────────────────────────
MAC_ADDR = "omni-worker"

from sqlalchemy import text as _text

def stats_init(conn):
    conn.execute(_text("""
        INSERT INTO worker_stats (mac_address, processed_ok, processed_err, pending, last_update, current_img)
        VALUES (:m, 0, 0, 0, now(), NULL)
        ON CONFLICT (mac_address) DO NOTHING;
    """), dict(m=MAC_ADDR))

# 其餘 stats_*、claim_tasks、update_capture_done、mark_error 與原版一致
# …

# ───────────────────────────
# Worker thread – batch version
# ───────────────────────────

def handle_rows(rows) -> List[Tuple[int, str, str, List[str]]]:
    """批量檢查雜湊 + 推論，回傳 (cid, img_path, sha_now, parsed) 列表"""
    paths: List[Path] = []
    cids: List[int] = []
    db_paths: List[str] = []

    for row in rows:
        local = db_to_local(row["img_path"])
        if not local.exists():
            raise FileNotFoundError(local)
        sha_now = sha256_file(local)
        sha_db = row["sha256_img"] or ""
        if sha_db and sha_db != sha_now:
            raise ValueError("sha256_img mismatch")
        paths.append(local)
        cids.append(row["id"])
        db_paths.append(row["img_path"])

    parsed_batch = omni_parse_json_batch(paths)
    return [(cid, db_p, sha, parsed) for cid, db_p, sha, parsed in zip(
        cids, db_paths, map(sha256_file, paths), parsed_batch)]


def worker_loop():
    while True:
        rows = claim_tasks(BATCH_SIZE)
        if not rows:
            time.sleep(2)
            continue
        try:
            with engine.begin() as conn:
                stats_progress(conn, rows[0]["img_path"])
            batch_info = handle_rows(rows)
            with engine.begin() as conn:
                for cid, _db_p, sha_now, parsed in batch_info:
                    update_capture_done(conn, cid, json.dumps(parsed, ensure_ascii=False), sha_now)
                stats_done(conn, ok=len(batch_info))
            logging.info("DONE ids=%s", ",".join(str(r[0]) for r in batch_info))
        except Exception:
            logging.exception("ERR batch (ids=%s)", ",".join(str(r["id"]) for r in rows))
            with engine.begin() as conn:
                for r in rows:
                    mark_error(conn, r["id"])
                stats_done(conn, err=len(rows))
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                paddle.device.cuda.empty_cache()

# ───────────────────────────
# Boot
# ───────────────────────────
if __name__ == "__main__":
    with engine.begin() as conn:
        stats_init(conn)
    logging.info("Worker started (threads=%d, batch=%d, device=%s)", THREADS, BATCH_SIZE, DEVICE)
    for _ in range(THREADS):
        threading.Thread(target=worker_loop, daemon=True).start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Shutdown – bye")
