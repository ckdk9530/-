#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniParser Worker – DB driven, JSON only  (v16 – model subprocesses)
================================================================
* v12: 修正 get_yolo_model 無 device 參數
* v13: 移除 get_som_labeled_img 的 yolo_result
* v14: 在 --debug-dir 模式新增統計 (count / total / avg / min / max)
* v15: --debug-dir 執行時改為輸出進度，不列印 JSON 內容
* v16: 三個模型改為子進程常駐服務
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
import atexit
import multiprocessing as mp
import uuid

from util.model_service import ensure_spawn_start_method


from util.memory import (
    debug_gpu_memory,
    maybe_empty_gpu_cache,
)

import torch
import gc
from PIL import Image
from util.prefetch import ImagePrefetcher

# ───────────────────────────
# CLI
# ───────────────────────────
def _parse_args():
    cli = argparse.ArgumentParser()
    cli.add_argument("--debug", action="store_true", help="單張 debug")
    cli.add_argument("--img", help="單張路徑")
    cli.add_argument("--debug-dir", help="資料夾批量 debug")
    cli.add_argument("--batch-size", type=int, default=8, help="一次 GPU 推論張數")
    cli.add_argument(
        "--no-print-text",
        dest="print_text",
        action="store_false",
        help="不在第一批結束後輸出文字列表",
    )
    cli.set_defaults(print_text=True)
    return cli.parse_args()

args = None  # type: argparse.Namespace | None
DEBUG = False
DEBUG_IMG: str | None = None
DEBUG_DIR: str | None = None
BATCH_SIZE = 8
PRINT_TEXT = True

# ───────────────────────────
# Model utils
# ───────────────────────────
from OmniParser.util.utils import get_som_labeled_img
from util.model_service import YoloWorker, OCRWorker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo_proc: YoloWorker | None = None
ocr_proc: OCRWorker | None = None

def _init_model_processes():
    global yolo_proc, ocr_proc
    yolo_proc = YoloWorker("weights/icon_detect/model.pt", device=DEVICE)
    ocr_proc = OCRWorker(use_gpu=(DEVICE == "cuda"))
    for p in (yolo_proc, ocr_proc):
        p.start()
    atexit.register(_shutdown_processes)

def _shutdown_processes():
    for p in (yolo_proc, ocr_proc):
        if p is not None:
            p.shutdown()


def _queue_ocr(path, use_paddleocr=True):
    """透過子進程執行 OCR"""
    assert ocr_proc is not None
    ocr_proc.in_q.put((0, str(path)))
    _idx, result = ocr_proc.out_q.get()
    return result

# ───────────────────────────
# Inference helpers
# ───────────────────────────
@torch.inference_mode()
def omni_parse_json_single(
    image_path: Path | str,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.10,
    imgsz: int = 640,
):
    return omni_parse_json_batch([Path(image_path)], box_threshold, iou_threshold, imgsz=imgsz)[0]


@torch.inference_mode()
def omni_parse_json_batch(
    image_paths: List[Path | str],
    box_threshold: float = 0.05,
    iou_threshold: float = 0.10,
    imgsz: int = 640,
) -> List[List[str]]:
    paths = [str(p) for p in image_paths]

    # --- YOLO ---
    assert yolo_proc is not None
    yolo_proc.in_q.put(paths)
    yolo_results = yolo_proc.out_q.get()

    # --- OCR ---
    assert ocr_proc is not None
    for idx, p in enumerate(paths):
        ocr_proc.in_q.put((idx, p))
    ocr_results = [None] * len(paths)
    for _ in paths:
        idx, res = ocr_proc.out_q.get()
        ocr_results[idx] = res[0][0]

    return ocr_results


def sec_to_hms(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string; if under one second show ms."""
    if seconds < 1:
        return f"{seconds:.3f}s"
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

# ───────────────────────────
# Debug shortcuts
# ───────────────────────────
def _run_debug():
    if DEBUG and not DEBUG_DIR:
        p = Path(DEBUG_IMG) if DEBUG_IMG else Path(input("Image path » ").strip())
        print(json.dumps(omni_parse_json_single(p), ensure_ascii=False, indent=2))
        return True

    if DEBUG_DIR:
        imgs = sorted(
            [p for p in Path(DEBUG_DIR).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        )[:5]
        print(f"Total images (debug first 5): {len(imgs)}")
        start_all = time.perf_counter()
        durations: List[float] = []
        processed = 0

        for i in range(0, len(imgs), BATCH_SIZE):
            batch = imgs[i : i + BATCH_SIZE]
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                results = omni_parse_json_batch(batch)
            t1 = time.perf_counter()

            save_debug_results([Path(p) for p in batch], results)

            if PRINT_TEXT and i == 0:
                print("First batch text:")
                for t in results:
                    print(t)

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
        return True
    return False

# ───────────────────────────
# Normal-mode Config & DB deps
# ───────────────────────────
LOG_LEVEL  = "INFO"
THREADS    = 4           # DB 輪詢 / commit 執行緒數
# BATCH_SIZE 由 CLI 參數決定 (預設 8)

from sqlalchemy.engine import URL

DB_URL = URL.create(
    drivername="postgresql+psycopg2",
    username="omniapp",
    password="uG6#9jT!wZ8rL@p2$Xv4qS1e%Nb0Ka7C",
    host="127.0.0.1",
    port=5432,
    database="screencap",
)

DB_PREFIX    = Path("/volume1/ScreenshotService")
LOCAL_PREFIX = Path("/mnt/nas/inbox")

from sqlalchemy import create_engine  # noqa: E402

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
engine = create_engine(DB_URL, pool_size=THREADS * 2, max_overflow=0)

# 影像預讀快取，容量與分批大小一致
PREFETCHER: ImagePrefetcher | None = None

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


def save_debug_results(paths: List[Path], texts: List[List[str]]) -> None:
    """將 debug 模式結果寫入資料庫"""

    with engine.begin() as conn:
        for p, txt in zip(paths, texts):
            conn.execute(
                _text(
                    """
                INSERT INTO captures (timestamp, img_path, sha256_img, status, json_payload)
                VALUES (now(), :p, :s, 'done', :j)
                ON CONFLICT (img_path) DO UPDATE
                    SET sha256_img = EXCLUDED.sha256_img,
                        status = 'done',
                        json_payload = EXCLUDED.json_payload;
                """
                ),
                dict(p=str(p), s=sha256_file(p), j=json.dumps(txt, ensure_ascii=False)),
            )

# ───────────────────────────
# worker_stats helpers (unchanged)
# ───────────────────────────
MAC_ADDR = "omni-worker"
RUN_ID = uuid.uuid4().hex

# ... (stats_* functions 保持原樣) ...
from sqlalchemy import text as _text

def stats_init(conn):
    conn.execute(
        _text(
            """
        INSERT INTO worker_stats (mac_address, run_id, processed_ok, processed_err, pending, last_update, current_img)
        SELECT :m, :r, 0, 0, 0, now(), NULL
        WHERE NOT EXISTS (
            SELECT 1 FROM worker_stats WHERE mac_address = :m
        );
        """
        ),
        dict(m=MAC_ADDR, r=RUN_ID),
    )

# 其餘 stats_*、claim_tasks、update_capture_done、mark_error 與原版一致
# 為節省篇幅，若未顯示請從原檔複製；或確保函式簽名不變。


def stats_progress(conn, img_path: str) -> None:
    """更新當前處理中的圖片路徑"""

    conn.execute(
        _text(
            """
        UPDATE worker_stats
           SET current_img = :p,
               last_update = now()
         WHERE mac_address = :m;
        """
        ),
        dict(p=img_path, m=MAC_ADDR),
    )


def stats_done(conn, ok: int = 0, err: int = 0) -> None:
    """累積完成數量"""

    conn.execute(
        _text(
            """
        UPDATE worker_stats
           SET processed_ok  = processed_ok  + :ok,
               processed_err = processed_err + :err,
               current_img   = NULL,
               last_update   = now()
         WHERE mac_address = :m;
        """
        ),
        dict(ok=ok, err=err, m=MAC_ADDR),
    )


def claim_tasks(n: int) -> list[dict]:
    """取得待處理任務並預讀圖片"""

    if n <= 0:
        return []

    with engine.begin() as conn:
        rows = conn.execute(
            _text(
                """
            UPDATE captures SET status = 'processing'
             WHERE id IN (
                SELECT id FROM captures
                 WHERE status = 'pending'
                 ORDER BY id
                 FOR UPDATE SKIP LOCKED
                 LIMIT :n
             )
            RETURNING id, img_path, sha256_img;
            """
            ),
            dict(n=n),
        ).mappings().all()

    tasks = [dict(r) for r in rows]

    if tasks and PREFETCHER is not None:
        PREFETCHER.prefetch(db_to_local(t["img_path"]) for t in tasks)

    return tasks


def update_capture_done(conn, cid: int, json_payload: str, sha_now: str) -> None:
    """將解析結果寫回資料庫"""

    conn.execute(
        _text(
            """
        UPDATE captures
           SET status      = 'done',
               sha256_img  = :s,
               json_payload = :j
         WHERE id = :cid;
        """
        ),
        dict(cid=cid, j=json_payload, s=sha_now),
    )


def mark_error(conn, cid: int) -> None:
    """標記任務為錯誤"""

    conn.execute(
        _text("UPDATE captures SET status = 'error' WHERE id = :cid"),
        dict(cid=cid),
    )

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

        assert PREFETCHER is not None
        sha_now = PREFETCHER.get_sha(local)
        _ = PREFETCHER.pop_image(local)
        if sha_now is None:
            sha_now = sha256_file(local)
        sha_db = row["sha256_img"] or ""
        if sha_db and sha_db != sha_now:
            raise ValueError("sha256_img mismatch")
        paths.append(local)
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
        if len(task_queue) < BATCH_SIZE:
            new_rows = claim_tasks(BATCH_SIZE - len(task_queue))
            if new_rows:
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
                maybe_empty_gpu_cache()
                debug_gpu_memory("worker_loop finally")

# ───────────────────────────
# Boot
# ───────────────────────────
def main() -> None:
    global args, DEBUG, DEBUG_IMG, DEBUG_DIR, BATCH_SIZE, PREFETCHER, PRINT_TEXT

    ensure_spawn_start_method()

    args = _parse_args()
    DEBUG = args.debug or bool(args.debug_dir)
    DEBUG_IMG = args.img
    DEBUG_DIR = args.debug_dir
    BATCH_SIZE = args.batch_size
    PRINT_TEXT = args.print_text

    PREFETCHER = ImagePrefetcher(size=BATCH_SIZE)

    _init_model_processes()

    if _run_debug():
        return

    with engine.begin() as conn:
        stats_init(conn)
    logging.info(
        "Worker started (threads=%d, batch=%d, device=%s)",
        THREADS,
        BATCH_SIZE,
        DEVICE,
    )
    for _ in range(THREADS):
        threading.Thread(target=worker_loop, daemon=True).start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logging.info("Shutdown – bye")


if __name__ == "__main__":
    main()
