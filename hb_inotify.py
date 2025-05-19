#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
"""
hb_inotify.py — single-thread heartbeat & DHCP unmatched scanner
  • Python 管理 timeout；RouterOS 只增刪名單
  • 一條迴圈：inotify → timeout → DHCP 掃描 → 5 min resync
  • 不再列印 HB-REFRESH
"""

import os, re, sys, fcntl, subprocess, time, pathlib
from collections import defaultdict
from datetime import datetime, date
from typing import Optional

import pyinotify
from routeros_api import RouterOsApiPool

# ────────── 全局設定 ──────────
ROOT = '/volume1/ScreenshotService'
ALIVE_LIST, BLOCK_LIST = 'internet-alive', 'internet-block'
HB_TIMEOUT, EXPIRE_IVL  = 30, 2          # 秒
DHCP_SCAN_SEC, RESYNC_SEC = 60, 300      # 秒

START_IP, END_IP = 40, 200
EXCLUDED_IPS = {
    '192.168.1.43', '192.168.1.80', '192.168.1.59', '192.168.1.136',
    '192.168.1.41', '192.168.1.153', '192.168.1.42', '192.168.1.44',
    '192.168.1.53', '192.168.1.60', '192.168.1.73', '192.168.1.171',
    '192.168.1.105', '192.168.1.79', '192.168.1.78', '192.168.1.63',
    '192.168.1.81', '192.168.1.50',
}                    # 與前相同

ROS_HOST, ROS_USER, ROS_PASS = '192.168.1.1', 'nas-api', 'MySTRONG!Pass123'
API_PORT, USE_SSL, PLAIN_LOGIN = 8728, False, True

# ────────── 單例鎖 ──────────
fd = os.open('/tmp/hb_inotify.lock', os.O_RDWR | os.O_CREAT)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print('[EXIT] another instance'); sys.exit(0)

# ────────── RouterOS 連線 ──────────
api_pool = RouterOsApiPool(ROS_HOST, ROS_USER, ROS_PASS,
                           port=API_PORT, use_ssl=USE_SSL,
                           plaintext_login=PLAIN_LOGIN)
api       = api_pool.get_api()
alist     = api.get_resource('/ip/firewall/address-list')
leases    = api.get_resource('/ip/dhcp-server/lease')

# ────────── 本地集合 ──────────
alive_set = {r['address'] for r in alist.get(list=ALIVE_LIST)}
block_set = {r['address'] for r in alist.get(list=BLOCK_LIST)}
last_seen = defaultdict(lambda: 0.0)

# ────────── 工具 ──────────
MAC_RE  = re.compile(r'^.+_[0-9A-Fa-f]{12}$')
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
EXF     = pyinotify.ExcludeFilter([r'/@eaDir/'])

def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')

def mac_to_ip(mac: str) -> Optional[str]:
    tgt = mac.lower()
    try:
        out = subprocess.check_output('arp -an', shell=True, text=True)
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 4 and p[3] != '<incomplete>' and \
           p[3].replace(':', '').lower() == tgt:
            return p[1].strip('()')
    return None

def remove_entry(rec):
    rid = rec.get('.id') or rec.get('id')
    if rid:
        alist.remove(id=rid)
    else:
        alist.remove(**{'list': rec['list'], 'address': rec['address']})

def sync_block(ip: str, reason: str):
    if ip in alive_set:
        for r in alist.get(list=ALIVE_LIST, address=ip):
            remove_entry(r)
        alive_set.discard(ip)
    if ip not in block_set:
        alist.add(list=BLOCK_LIST, address=ip,
                  comment=f'{reason} {ip} {now_iso()}')
        block_set.add(ip)
    last_seen.pop(ip, None)
    print(f'[BLOCK-{reason}]', ip)

# ────────── inotify handlers ──────────
class DateWatcher(pyinotify.ProcessEvent):
    def __init__(self, wm): self.wm = wm
    def process_IN_CREATE(self, ev):
        if ev.dir and DATE_RE.fullmatch(os.path.basename(ev.pathname)):
            if os.path.basename(ev.pathname) == date.today().strftime('%Y-%m-%d'):
                print('[WATCH-DATE]', ev.pathname)
                self.wm.add_watch(ev.pathname, pyinotify.IN_CLOSE_WRITE,
                                  rec=False, quiet=True, exclude_filter=EXF)

class UserWatcher(pyinotify.ProcessEvent):
    def __init__(self, wm): self.wm = wm
    def process_IN_CREATE(self, ev):
        if ev.dir and MAC_RE.fullmatch(os.path.basename(ev.pathname)):
            print('[WATCH-USER]', ev.pathname)
            add_user_watch(ev.pathname)

class ShotHandler(pyinotify.ProcessEvent):
    def process_IN_CLOSE_WRITE(self, ev):
        if not ev.pathname.lower().endswith(('.jpg', '.png')): return
        mac = ev.pathname.split(os.sep)[-3].split('_')[-1]
        ip  = mac_to_ip(mac)
        if not ip:
            print('[MISS-ARP]', mac); return
        now = time.time()
        if now - last_seen[ip] < 4: return
        last_seen[ip] = now
        if ip in alive_set:
            return                      # 刷新但不打印
        if ip in block_set:
            for r in alist.get(list=BLOCK_LIST, address=ip):
                remove_entry(r)
            block_set.remove(ip)
        alist.add(list=ALIVE_LIST, address=ip,
                  comment=f'HB {ip} {now_iso()}')
        alive_set.add(ip)
        print('[HB-ADD]', ip)

# ────────── watch 管理 ──────────
def add_user_watch(path):
    wm.add_watch(path, pyinotify.IN_CREATE, rec=False, quiet=True,
                 proc_fun=DateWatcher(wm), exclude_filter=EXF)
    today_dir = os.path.join(path, date.today().strftime('%Y-%m-%d'))
    if os.path.isdir(today_dir):
        print('[WATCH-DATE]', today_dir)
        wm.add_watch(today_dir, pyinotify.IN_CLOSE_WRITE, rec=False,
                     quiet=True, exclude_filter=EXF)

# ────────── DHCP 比對 ──────────
def todays_mac_set() -> set[str]:
    macs = set()
    today = date.today().strftime('%Y-%m-%d')
    for p in pathlib.Path(ROOT).iterdir():
        if p.is_dir() and MAC_RE.fullmatch(p.name) and (p / today).is_dir():
            macs.add(p.name.split('_')[-1].lower())
    return macs

def scan_unmatched():
    try:
        lss = leases.get()
    except Exception as e:
        print('[ERR-DHCP]', e); return
    today_macs = todays_mac_set()
    for l in lss:
        ip = l.get('address')
        if (not ip or ip in EXCLUDED_IPS or l.get('status') != 'bound'):
            continue
        if not (START_IP <= int(ip.split('.')[-1]) <= END_IP): continue
        mac = l.get('mac-address', '').replace(':', '').lower()
        if mac and mac not in today_macs:
            sync_block(ip, 'UNMATCH')

# ────────── 初始化 inotify ──────────
wm = pyinotify.WatchManager()
notifier = pyinotify.Notifier(wm, default_proc_fun=ShotHandler())
wm.add_watch(ROOT, pyinotify.IN_CREATE, rec=False, quiet=True,
             proc_fun=UserWatcher(wm), exclude_filter=EXF)
for d in os.scandir(ROOT):
    if d.is_dir() and MAC_RE.fullmatch(d.name):
        print('[WATCH-USER]', d.path)
        add_user_watch(d.path)

print('[INIT] single-thread watcher running')

# ────────── 定時器 ──────────
next_exp  = time.time() + EXPIRE_IVL
next_scan = time.time() + DHCP_SCAN_SEC
next_sync = time.time() + RESYNC_SEC

# ────────── 主迴圈 ──────────
while True:
    timeout = min(next_exp, next_scan, next_sync) - time.time()
    notifier.process_events()
    if notifier.check_events(max(timeout, 0)):
        notifier.read_events()

    now = time.time()

    if now >= next_exp:                 # 心跳逾時
        for ip, ts in list(last_seen.items()):
            if now - ts > HB_TIMEOUT:
                sync_block(ip, 'TIMEOUT')
        next_exp = now + EXPIRE_IVL

    if now >= next_scan:                # DHCP 比對
        scan_unmatched()
        next_scan = now + DHCP_SCAN_SEC

    if now >= next_sync:                # 5 min resync
        alive_set = {r['address'] for r in alist.get(list=ALIVE_LIST)}
        block_set = {r['address'] for r in alist.get(list=BLOCK_LIST)}
        for ip in list(last_seen):
            if ip not in alive_set:
                last_seen.pop(ip, None)
        print('[RESYNC] refreshed sets from RouterOS')
        next_sync = now + RESYNC_SEC
