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
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import os
os.environ["FLAGS_allocator_strategy"] = "auto_growth"  # GPU 記憶體按需配置

import paddle
import contextlib
import io
import atexit
import uuid

from util.model_service import ensure_spawn_start_method


from util.memory import (
    debug_gpu_memory,
    maybe_empty_gpu_cache,
)

import torch
import gc
from util.prefetch import ImagePrefetcher

# ───────────────────────────
# CLI
# ───────────────────────────
def _parse_args():
    cli = argparse.ArgumentParser()
    cli.add_argument("--debug", action="store_true", help="單張 debug")
    cli.add_argument("--img", help="單張路徑")
    cli.add_argument("--debug-dir", help="資料夾批量 debug")
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
PRINT_TEXT = True


# ───────────────────────────
# Model utils
# ───────────────────────────
from OmniParser.util.utils import (
    get_som_labeled_img,
    get_caption_model_processor,
)
from util.model_service import YoloWorker, OCRWorker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

yolo_proc: YoloWorker | None = None
ocr_proc: OCRWorker | None = None
caption_model_processor = None

def _init_model_processes():
    global yolo_proc, ocr_proc, caption_model_processor
    yolo_proc = YoloWorker("weights/icon_detect/model.pt", device=DEVICE)
    ocr_proc = OCRWorker(use_gpu=(DEVICE == "cuda"))
    caption_model_processor = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path="weights/icon_caption_florence",
        device=DEVICE,
    )
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
) -> List[List[dict]]:
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
        ocr_results[idx] = res[0]

    outputs: List[List[dict]] = []
    for path, yolo_res, (ocr_text, ocr_bbox) in zip(paths, yolo_results, ocr_results):
        _, _coords, parsed = get_som_labeled_img(
            path,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            caption_model_processor=caption_model_processor,
            ocr_text=ocr_text,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            yolo_result=yolo_res,
        )
        outputs.append(parsed)

    return outputs


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

        for img in imgs:
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                result = omni_parse_json_single(img)
            t1 = time.perf_counter()

            save_debug_results([Path(img)], [result])

            if PRINT_TEXT and processed == 0:
                print("First text:")
                print(result)

            per_img = t1 - t0
            durations.append(per_img)

            processed += 1
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
engine = create_engine(DB_URL, pool_size=2, max_overflow=0)

# 影像預讀快取
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

def save_debug_results(paths: List[Path], texts: List[List[str]]) -> None:
    """將 debug 模式結果寫入資料庫"""

    with engine.begin() as conn:
        for p, txt in zip(paths, texts):
            conn.execute(
                _text(
                    """
                INSERT INTO captures (timestamp, img_path, status, json_payload)
                VALUES (now(), :p, 'done', :j)
                ON CONFLICT (img_path) DO UPDATE
                    SET status = 'done',
                        json_payload = EXCLUDED.json_payload;
                """
                ),
                dict(p=str(p), j=json.dumps(txt, ensure_ascii=False)),
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
            RETURNING id, img_path;
            """
            ),
            dict(n=n),
        ).mappings().all()

    tasks = [dict(r) for r in rows]

    if tasks and PREFETCHER is not None:
        PREFETCHER.prefetch(db_to_local(t["img_path"]) for t in tasks)

    return tasks


def update_capture_done(conn, cid: int, json_payload: str) -> None:
    """將解析結果寫回資料庫"""

    conn.execute(
        _text(
            """
        UPDATE captures
           SET status      = 'done',
               json_payload = :j
         WHERE id = :cid;
        """
        ),
        dict(cid=cid, j=json_payload),
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

def handle_row(row) -> Tuple[int, str, List[dict]]:
    """檢查雜湊並解析單張圖片"""
    local = db_to_local(row["img_path"])
    if not local.exists():
        raise FileNotFoundError(local)

    assert PREFETCHER is not None
    sha_now: str | None = None
    if not SKIP_SHA256_CHECK:
        sha_now = PREFETCHER.get_sha(local)
        _ = PREFETCHER.pop_image(local)
        if sha_now is None:
            sha_now = sha256_file(local.read_bytes())
        sha_db = row["sha256_img"] or ""
        if sha_db and sha_db != sha_now:
            raise ValueError("sha256_img mismatch")
    else:
        _ = PREFETCHER.pop_image(local)

    parsed = omni_parse_json_single(local)
    return row["id"], row["img_path"], parsed


def worker_loop():
    while True:
        rows = claim_tasks(1)
        if not rows:
            time.sleep(2)
            continue
        row = rows[0]
        try:
            with engine.begin() as conn:
                stats_progress(conn, row["img_path"])
            info = handle_row(row)
            with engine.begin() as conn:
                cid, _db_p, parsed = info
                update_capture_done(conn, cid, json.dumps(parsed, ensure_ascii=False))
                stats_done(conn, ok=1)
            logging.info("DONE id=%s", info[0])
        except Exception:
            logging.exception("ERR id=%s", row["id"])
            with engine.begin() as conn:
                mark_error(conn, row["id"])
                stats_done(conn, err=1)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                maybe_empty_gpu_cache()
                debug_gpu_memory("worker_loop finally")

# ───────────────────────────
# Boot
# ───────────────────────────
def main() -> None:
    global args, DEBUG, DEBUG_IMG, DEBUG_DIR, PREFETCHER, PRINT_TEXT, SKIP_SHA256_CHECK

    ensure_spawn_start_method()

    args = _parse_args()
    DEBUG = args.debug or bool(args.debug_dir)
    DEBUG_IMG = args.img
    DEBUG_DIR = args.debug_dir
    PRINT_TEXT = args.print_text
    SKIP_SHA256_CHECK = args.skip_sha256_check

    PREFETCHER = ImagePrefetcher(size=1, calc_sha=not SKIP_SHA256_CHECK)

    _init_model_processes()

    if _run_debug():
        return

    with engine.begin() as conn:
        stats_init(conn)
    logging.info(
        "Worker started (device=%s)",
        DEVICE,
    )
    try:
        worker_loop()
    except KeyboardInterrupt:
        logging.info("Shutdown – bye")


if __name__ == "__main__":
    main()
