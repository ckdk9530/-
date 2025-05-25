#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniParser Worker – DB driven, JSON only  (v15 – debug progress)
================================================================
* v12: 修正 get_yolo_model 無 device 參數
* v13: 移除 get_som_labeled_img 的 yolo_result
* v14: 在 --debug-dir 模式新增統計 (count / total / avg / min / max)
* v15: --debug-dir 執行時改為輸出進度，不列印 JSON 內容
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

import os
os.environ["FLAGS_allocator_strategy"] = "auto_growth"  # GPU 記憶體按需配置

import paddle
import contextlib
import io
from queue import Queue

from util.memory import debug_gpu_memory, release_ocr_gpu_cache

import torch
import gc
from PIL import Image
from util.prefetch import ImagePrefetcher

# ───────────────────────────
# CLI
# ───────────────────────────
cli = argparse.ArgumentParser()
cli.add_argument("--debug", action="store_true", help="單張 debug")
cli.add_argument("--img", help="單張路徑")
cli.add_argument("--debug-dir", help="資料夾批量 debug")
cli.add_argument("--batch-size", type=int, default=8, help="一次 GPU 推論張數")
cli.add_argument("--ocr-worker", "--ocr_worker", type=int, default=1, help="OCR 執行緒數量")
args, _ = cli.parse_known_args()
DEBUG = args.debug or bool(args.debug_dir)
DEBUG_IMG: str | None = args.img
DEBUG_DIR: str | None = args.debug_dir
BATCH_SIZE = args.batch_size
OCR_WORKERS = args.ocr_worker

# ───────────────────────────
# Model utils
# ───────────────────────────
from OmniParser.util.utils import (
    check_ocr_box,
    get_caption_model_processor,
    get_som_labeled_img,
    get_yolo_model,
    _get_paddle_ocr,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = get_yolo_model("weights/icon_detect/model.pt", device=DEVICE)
caption_model_processor = get_caption_model_processor(
    model_name="florence2",
    model_name_or_path="weights/icon_caption_florence",
    device=DEVICE,
)

# OCR 實例列表，供釋放 GPU 快取使用
_OCR_INSTANCES: list = []

# ───────────────────────────
# OCR worker thread & queue
# ───────────────────────────
OCR_QUEUE: "Queue" = Queue()

def _ocr_worker():
    # 每個執行緒建立獨立的 PaddleOCR 實例，避免併發競爭
    paddle_ocr = _get_paddle_ocr.__wrapped__(use_gpu=(DEVICE == "cuda"))
    _OCR_INSTANCES.append(paddle_ocr)
    while True:
        img, res_q, use_paddle = OCR_QUEUE.get()
        if img is None:
            break
        try:
            result = check_ocr_box(
                img,
                output_bb_format="xyxy",
                use_paddleocr=use_paddle,
                paddle_ocr=paddle_ocr if use_paddle else None,
                easyocr_args={"paragraph": False, "text_threshold": 0.9},
            )
        except Exception as e:
            result = e
        res_q.put(result)

_OCR_THREADS = []
for _ in range(OCR_WORKERS):
    t = threading.Thread(target=_ocr_worker, daemon=True)
    t.start()
    _OCR_THREADS.append(t)

def _queue_ocr(img, use_paddleocr=True):
    q = Queue()
    OCR_QUEUE.put((img, q, use_paddleocr))
    result = q.get()
    if isinstance(result, Exception):
        raise result
    return result

# ───────────────────────────
# Inference helpers
# ───────────────────────────
@torch.inference_mode()
def omni_parse_json_single(
    image_path: Path | Image.Image,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.10,
    use_paddleocr: bool = True,
    imgsz: int = 640,
):
    if isinstance(image_path, Image.Image):
        img = image_path.convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")
    (ocr_text, ocr_bbox), _ = _queue_ocr(img, use_paddleocr)
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
    image_paths: List[Path | Image.Image],
    box_threshold: float = 0.05,
    iou_threshold: float = 0.10,
    use_paddleocr: bool = True,
    imgsz: int = 640,
) -> List[List[str]]:
    imgs = [
        (p.convert("RGB") if isinstance(p, Image.Image) else Image.open(p).convert("RGB"))
        for p in image_paths
    ]
    _ = yolo_model.predict(imgs, batch=len(imgs), verbose=False)

    # --- OCR ---
    # 一次將整批影像送入佇列，讓多個 _ocr_worker 平行處理
    q_list = []
    for img in imgs:
        q = Queue()
        OCR_QUEUE.put((img, q, use_paddleocr))
        q_list.append(q)

    ocr_results = []
    for q in q_list:
        result = q.get()
        if isinstance(result, Exception):
            raise result
        ocr_results.append(result[0])

    outputs: List[List[str]] = []
    for img, (ocr_text, ocr_bbox) in zip(imgs, ocr_results):
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
        for inst in _OCR_INSTANCES:
            release_ocr_gpu_cache(inst)
        debug_gpu_memory("omni_parse_json_batch")

    return outputs


def sec_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# ───────────────────────────
# Debug shortcuts
# ───────────────────────────
if DEBUG and not DEBUG_DIR:
    p = Path(DEBUG_IMG) if DEBUG_IMG else Path(input("Image path » ").strip())
    print(json.dumps(omni_parse_json_single(p), ensure_ascii=False, indent=2))
    sys.exit()

if DEBUG_DIR:
    imgs = sorted(
        [p for p in Path(DEBUG_DIR).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )
    print(f"Total images: {len(imgs)}")
    start_all = time.perf_counter()
    durations: List[float] = []
    processed = 0

    for i in range(0, len(imgs), BATCH_SIZE):
        batch = imgs[i : i + BATCH_SIZE]
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            _ = omni_parse_json_batch(batch)
        t1 = time.perf_counter()

        per_img = (t1 - t0) / len(batch)
        durations.extend([per_img] * len(batch))

        processed += len(batch)
        elapsed = time.perf_counter() - start_all
        avg_so_far = sum(durations) / len(durations)
        eta = max(len(imgs) - processed, 0) * avg_so_far
        print(
            f"{processed}/{len(imgs)} {per_img:.3f}s "
            f"(elapsed {sec_to_hms(elapsed)}, eta {sec_to_hms(eta)})"
        )

    total_time = time.perf_counter() - start_all
    print("\n=== Summary ===")
    print(f"Images : {len(imgs)}")
    print(f"Total  : {sec_to_hms(total_time)}")
    if imgs:
        print(f"Avg    : {sec_to_hms(total_time / len(imgs))}")
        print(f"Min    : {sec_to_hms(min(durations))}")
        print(f"Max    : {sec_to_hms(max(durations))}")
    sys.exit()

# ───────────────────────────
# Normal-mode Config & DB deps
# ───────────────────────────
LOG_LEVEL  = "INFO"
THREADS    = 4           # DB 輪詢 / commit 執行緒數
BATCH_SIZE = 8           # ➜ GPU 一次推論張數 / DB claim

DB_URL = (
    "postgresql+psycopg2://omniapp:"
    "uG6#9jT!wZ8rL@p2$Xv4qS1e%Nb0Ka7C"
    "@127.0.0.1:5432/screencap"
)

DB_PREFIX    = Path("/volume1/ScreenshotService")
LOCAL_PREFIX = Path("/mnt/nas/inbox")

from sqlalchemy import create_engine, text  # noqa: E402

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
engine = create_engine(DB_URL, pool_size=THREADS * 2, max_overflow=0)

# 影像預讀快取，容量為兩個分批大小
PREFETCHER = ImagePrefetcher(size=BATCH_SIZE * 2)

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

# ... (stats_* functions 保持原樣) ...
from sqlalchemy import text as _text

def stats_init(conn):
    conn.execute(_text("""
        INSERT INTO worker_stats (mac_address, processed_ok, processed_err, pending, last_update, current_img)
        VALUES (:m, 0, 0, 0, now(), NULL)
        ON CONFLICT (mac_address) DO NOTHING;
    """), dict(m=MAC_ADDR))

# 其餘 stats_*、claim_tasks、update_capture_done、mark_error 與原版一致
# 為節省篇幅，若未顯示請從原檔複製；或確保函式簽名不變。

# ───────────────────────────
# Worker thread – batch version
# ───────────────────────────

def handle_rows(rows) -> List[Tuple[int, str, str, List[str]]]:
    """批量檢查雜湊 + 推論，回傳 (cid, img_path, sha_now, parsed) 列表"""
    paths: List[Path | Image.Image] = []
    cids: List[int] = []
    db_paths: List[str] = []
    sha_list: List[str] = []

    for row in rows:
        local = db_to_local(row["img_path"])
        if not local.exists():
            raise FileNotFoundError(local)

        img = PREFETCHER.pop_image(local)
        sha_now = PREFETCHER.get_sha(local)
        if sha_now is None:
            sha_now = sha256_file(local)
        sha_db = row["sha256_img"] or ""
        if sha_db and sha_db != sha_now:
            raise ValueError("sha256_img mismatch")
        paths.append(img if img is not None else local)
        sha_list.append(sha_now)
        cids.append(row["id"])
        db_paths.append(row["img_path"])

    parsed_batch = omni_parse_json_batch(paths)
    return [
        (cid, db_p, sha, parsed)
        for cid, db_p, sha, parsed in zip(cids, db_paths, sha_list, parsed_batch)
    ]


def worker_loop():
    task_queue: list[dict] = []
    while True:
        if len(task_queue) < BATCH_SIZE * 2:
            new_rows = claim_tasks(BATCH_SIZE * 2 - len(task_queue))
            if new_rows:
                PREFETCHER.prefetch(db_to_local(r["img_path"]) for r in new_rows)
                task_queue.extend(new_rows)

        if len(task_queue) < BATCH_SIZE:
            if not task_queue:
                time.sleep(2)
            continue

        rows = [task_queue.pop(0) for _ in range(min(BATCH_SIZE, len(task_queue)))]
        try:
            with engine.begin() as conn:
                # 記錄目前批次第一張做進度即可
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
                debug_gpu_memory("worker_loop finally")

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
    finally:
        pass
