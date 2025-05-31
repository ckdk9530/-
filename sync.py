#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniParser – Incremental Sync (staging + per‑PC multithread + 120‑day retention)
==============================================================================
• 查最新拍攝日 → AUTOCOMMIT 短連線。
• 主流程：COPY → INSERT → DELETE → PURGE >120d。
• 啟用 `SET LOCAL synchronous_commit = off` 降低 WAL flush；COPY/INSERT 分批 commit (CHUNK=100k)。
"""
from __future__ import annotations
import re, io, tempfile, os
from pathlib import Path
from datetime import datetime, timedelta, date
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# ─── 常量 ─────────────────────────────────────────
ROOT_DIR   = Path("/volume1/ScreenshotService")
MAC_RE     = re.compile(r".+_[0-9a-f]{12}$", re.I)
DATE_RE    = re.compile(r"\d{4}-\d{2}-\d{2}")
DAYS_WINDOW = 90
DB_URL = URL.create(
    drivername="postgresql+psycopg2",
    username="omniapp",
    password="uG6#9jT!wZ8rL@p2$Xv4qS1e%Nb0Ka7C",
    host="192.168.1.251",
    port=5432,
    database="screencap",
)
POOL_SIZE    = 4
TMP_TABLE    = "captures_paths_tmp"
IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_WORKERS  = 10             # 建議值：CPU×2
LOCAL_TZ     = 'Asia/Shanghai'
CHUNK        = 100_000        # 每批 10 萬筆 COPY + COMMIT

# ─── 掃盤 ─────────────────────────────────────────

def _collect_paths_for_pc(pc_dir: Path, min_date: date) -> list[str]:
    """回傳指定電腦目錄下待同步檔案"""
    lst: list[str] = []
    for date_dir in pc_dir.iterdir():
        if (not date_dir.is_dir()) or (not DATE_RE.fullmatch(date_dir.name)):
            continue
        d = datetime.strptime(date_dir.name, "%Y-%m-%d").date()
        if d < min_date or d < (datetime.today().date() - timedelta(days=DAYS_WINDOW)):
            continue
        for img in date_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMG_SUFFIXES:
                lst.append(str(img))
    return lst

def scan_to_temp(tmp_file: io.TextIOBase, min_date: date) -> int:
    total = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        pc_dirs = [d for d in ROOT_DIR.iterdir() if d.is_dir() and MAC_RE.fullmatch(d.name)]
        futures = [exe.submit(_collect_paths_for_pc, d, min_date) for d in pc_dirs]
        for fut in as_completed(futures):
            for p in fut.result():
                tmp_file.write(f"{p}\n"); total += 1
    tmp_file.flush(); return total

# ─── 同步流程 ─────────────────────────────────────

def run_sync() -> None:
    t0 = time(); print("[START] sync …")
    # a) 最新拍攝日 (AUTOCOMMIT)
    with create_engine(DB_URL, isolation_level="AUTOCOMMIT").connect() as c:
        latest = c.execute(text("SELECT MAX(last_sync)::date FROM captures")).scalar()
    min_date = latest or (datetime.today().date() - timedelta(days=DAYS_WINDOW))
    print(f"[INFO] 增量掃描自 {min_date}")

    # b) 掃盤
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        total_files = scan_to_temp(tmp, min_date)
    print(f"[SCAN] {total_files:,} files → {tmp_path}")

    if total_files == 0:
        print("[DONE] 無新增檔案；耗時 %.2fs" % (time()-t0))
        os.remove(tmp_path)
        return

    # c) DB 寫入
    engine = create_engine(DB_URL, pool_size=POOL_SIZE, max_overflow=0, pool_pre_ping=True)
    conn = engine.raw_connection(); cur = conn.cursor()
    try:
        cur.execute(
            f"CREATE UNLOGGED TABLE IF NOT EXISTS {TMP_TABLE} (img_path text PRIMARY KEY);"
        )
        cur.execute(f"TRUNCATE {TMP_TABLE};")
        cur.execute("SET LOCAL synchronous_commit = off;")

        # COPY 分批
        with tmp_path.open() as f:
            while True:
                buf_lines = [f.readline() for _ in range(CHUNK)]
                buf_lines = [l for l in buf_lines if l]
                if not buf_lines:
                    break
                cur.copy_from(io.StringIO(''.join(buf_lines)), TMP_TABLE, columns=("img_path",))
                conn.commit()
        print("[COPY] 完成")

        # 唯一索引
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS captures_img_path_uniq ON captures(img_path);")

        # INSERT 新增路徑
        cur.execute(f"""
            INSERT INTO captures (
                timestamp, computer_name, mac_address, monitor_no,
                img_path, status, last_sync)
            SELECT (now() AT TIME ZONE '{LOCAL_TZ}'),
                   split_part(split_part(img_path,'/',4),'_',1),
                   lower(right(split_part(img_path,'/',4), 12)),
                   split_part(split_part(img_path, '_display_', 2), '.', 1)::int,
                   img_path,
                   'pending',
                   to_timestamp(regexp_replace(img_path,
                       '^.*screenshot_(\\d{{4}}-\\d{{2}}-\\d{{2}})_(\\d{{2}})-(\\d{{2}})-(\\d{{2}})_display_\\d+.*$',
                       '\\1 \\2:\\3:\\4'), 'YYYY-MM-DD HH24:MI:SS')
              FROM {TMP_TABLE}
            ON CONFLICT (img_path) DO UPDATE
                SET status = 'pending';""")
        print(f"[INSERT] +{cur.rowcount:,}")

        # DELETE vanished
        cur.execute(f"DELETE FROM captures c WHERE NOT EXISTS (SELECT 1 FROM {TMP_TABLE} t WHERE t.img_path=c.img_path);")
        print(f"[DELETE] -{cur.rowcount:,}")

        # PURGE >120d
        cur.execute(f"DELETE FROM captures WHERE last_sync < current_date - INTERVAL '{DAYS_WINDOW} days';")
        print(f"[PURGE] -{cur.rowcount:,}")

        conn.commit()
    finally:
        conn.close(); os.remove(tmp_path)

    print(f"[DONE] {time()-t0:.2f}s")

if __name__ == "__main__":
    run_sync()
