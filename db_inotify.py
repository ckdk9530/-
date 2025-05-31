#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""實時監控截圖目錄並將新增檔案寫入資料庫"""

import os
import re
from pathlib import Path

import pyinotify
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


# ─── 基本設定 ─────────────────────────────────────
ROOT_DIR = Path("/volume1/ScreenshotService")
DB_URL = URL.create(
    drivername="postgresql+psycopg2",
    username="omniapp",
    password="uG6#9jT!wZ8rL@p2$Xv4qS1e%Nb0Ka7C",
    host="192.168.1.251",
    port=5432,
    database="screencap",
)
LOCAL_TZ = "Asia/Shanghai"
IMG_SUFFIXES = (".jpg", ".jpeg", ".png")
MAC_RE = re.compile(r".+_[0-9A-Fa-f]{12}$")
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

engine = create_engine(DB_URL, pool_size=2, max_overflow=0)
EXF = pyinotify.ExcludeFilter([r"/@eaDir/"])

# ─── DB 寫入 ─────────────────────────────────────
INSERT_SQL = text(
    f"""
    INSERT INTO captures (
        timestamp, computer_name, mac_address, monitor_no,
        img_path, status, last_sync)
    VALUES (
        (now() AT TIME ZONE '{LOCAL_TZ}'),
        split_part(split_part(:p,'/',4),'_',1),
        lower(right(split_part(:p,'/',4), 12)),
        split_part(split_part(:p, '_display_', 2), '.', 1)::int,
        :p,
        'pending',
        to_timestamp(regexp_replace(:p,
            '^.*screenshot_(\\d{{4}}-\\d{{2}}-\\d{{2}})_(\\d{{2}})-(\\d{{2}})-(\\d{{2}})_display_\\d+.*$',
            '\\1 \\2:\\3:\\4'), 'YYYY-MM-DD HH24:MI:SS')
    )
    ON CONFLICT (img_path) DO UPDATE
        SET status = 'pending';
    """
)


def insert_capture(path: Path) -> None:
    with engine.begin() as conn:
        conn.execute(INSERT_SQL, dict(p=str(path)))


# ─── inotify handlers ─────────────────────────────────
class DateWatcher(pyinotify.ProcessEvent):
    def __init__(self, wm):
        self.wm = wm

    def process_IN_CREATE(self, ev):
        if ev.dir and DATE_RE.fullmatch(os.path.basename(ev.pathname)):
            print("[WATCH-DATE]", ev.pathname)
            self.wm.add_watch(
                ev.pathname,
                pyinotify.IN_CLOSE_WRITE,
                rec=False,
                quiet=True,
                exclude_filter=EXF,
            )


class UserWatcher(pyinotify.ProcessEvent):
    def __init__(self, wm):
        self.wm = wm

    def process_IN_CREATE(self, ev):
        if ev.dir and MAC_RE.fullmatch(os.path.basename(ev.pathname)):
            print("[WATCH-USER]", ev.pathname)
            add_user_watch(ev.pathname)


class ShotHandler(pyinotify.ProcessEvent):
    def process_IN_CLOSE_WRITE(self, ev):
        if not ev.pathname.lower().endswith(IMG_SUFFIXES):
            return
        try:
            insert_capture(Path(ev.pathname))
            print("[DB-INSERT]", ev.pathname)
        except Exception as e:
            print("[ERR-DB]", e, ev.pathname)


# ─── watch 管理 ─────────────────────────────────────
def add_user_watch(path: str) -> None:
    wm.add_watch(
        path,
        pyinotify.IN_CREATE,
        rec=False,
        quiet=True,
        proc_fun=DateWatcher(wm),
        exclude_filter=EXF,
    )
    for d in os.scandir(path):
        if d.is_dir() and DATE_RE.fullmatch(d.name):
            print("[WATCH-DATE]", d.path)
            wm.add_watch(
                d.path,
                pyinotify.IN_CLOSE_WRITE,
                rec=False,
                quiet=True,
                exclude_filter=EXF,
            )


# ─── 初始化 inotify ─────────────────────────────────
wm = pyinotify.WatchManager()
notifier = pyinotify.Notifier(wm, default_proc_fun=ShotHandler())
wm.add_watch(
    str(ROOT_DIR),
    pyinotify.IN_CREATE,
    rec=False,
    quiet=True,
    proc_fun=UserWatcher(wm),
    exclude_filter=EXF,
)
for d in os.scandir(ROOT_DIR):
    if d.is_dir() and MAC_RE.fullmatch(d.name):
        print("[WATCH-USER]", d.path)
        add_user_watch(d.path)

print("[INIT] db_inotify watcher running")

notifier.loop()
