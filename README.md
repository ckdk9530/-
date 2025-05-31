# 企业电脑管理系统

## 项目简介

企业电脑管理系统 (ScreenPipe) 旨在在不侵扰终端用户体验的前提下，持续收集员工电脑屏幕截图并进行 OCR / LLM 解析，以便于工单追踪、知识萃取及安全审计。系统由以下四大组件协同工作：

1. **Capture Agent** — 安装在员工电脑上的后台进程，每 5 秒自动截屏并上传。
2. **Synology NAS** — 集中存储原始截图，同时预留容器环境以承载后续 Web 前端。
3. **GPU Workstation** — 运行 OmniParser GPU worker、PostgreSQL 数据库以及自托管 AI‑Proxy。
4. **Web 前端（草稿）** — FastAPI + React Dashboard，部署在 NAS Docker 上，读取 Workstation 上的数据库并就地渲染 OCR / LLM 结果。

## 硬件与网络架构

```text
┌─────────────┐    FTP (TCP/21, passive)    ┌──────────────────────────┐
│ Employee PC │────────────────────────────▶│  Synology NAS            │
│ (Windows)   │                            │  • /volume1/screens       │
│ • Capture   │                            │  • Web‑UI (Docker, TODO) │
└─────────────┘                            └─────────────▲────────────┘
        ▲  SMB / NFS (readonly mount)                     │  (HTTP/443)
        │                                                 │
        │                                                 │
┌──────────────────────────────────────────────────────────┴────────────┐
│          GPU Workstation   192.168.1.240 (Ubuntu 22.04 + RTX 4090)    │
│  • PostgreSQL 16  (5432/TCP)                                          │
│  • OmniParser worker.py   (YOLOv8 + LayoutLMv3)                       │
│  • ai‑proxy (Cloudflare Workers 本地调试)                             │
└───────────────────────────────────────────────────────────────────────┘
```

| 设备角色                | 关键服务                                    | 典型配置/备注                          |
| ------------------- | --------------------------------------- | -------------------------------- |
| **Synology NAS**    | `vsftpd`, `docker`, <br>`web‑ui (计划)`   | DS1522+，16 TB RAID‑6             |
| **GPU Workstation** | `PostgreSQL`, `omni-worker`, `ai-proxy` | Ryzen 9 7950X，RTX 4090，64 GB RAM |
| **员工电脑 (Windows)**  | `capture_agent.py`                      | Python 3.10，双显示器                 |

> **带宽预算**：单台 PC 按 5 秒/屏 × 200 KB ≈ 40 KB/s，50 台同时上传峰值 < 2 MB/s，可由千兆 LAN 轻松承载。

## 代码仓库结构

```
.
├── amikon_network.py   # 屏幕抓取客戶端
├── sync.py             # NAS → PostgreSQL 同步
├── worker_batch.py     # GPU OCR 解析（已改為順序執行）
├── hb_inotify.py       # RouterOS 心跳監控
├── util/               # 共用工具模組
├── webui/              # FastAPI + 靜態頁範例
├── OmniParser/         # OmniParser 依賴
└── ...
```

### 1. 屏幕抓取客户端 (`amikon_network.py`)

* 依赖 `mss`, `Pillow`, `psutil`, `ftplib`。
* 每 5 秒截取所有显示器，命名为 `screenshot_YYYY-MM-DD_HH-MM-SS_display_N.jpg`，并按 `PCNAME_MAC/YYYY-MM-DD/` 结构上传至 NAS。
* 本地进程监控 RSS，不超过 500 MB 即自动重启。

### 2. NAS 同步服务 (`sync.py`)

* 通过 `sshfs` 或本地挂载读取 `/volume1/screens` 目录差异。
* 使用 `psycopg2.copy_from()` 将新文件路径批量导入 GPU Workstation 上的 `captures_paths_tmp`。
* 连接字符串示例：`postgres://omniapp:<pwd>@192.168.1.240:5432/omni`。
* 在 DSM「计划任务」中以 `python3 /volume1/Server_management/sync.py` 调度，输出记录可在「任务日志」查看。

### 3. GPU Workstation 解析流水 (`worker_batch.py`)

1. 轮询数据库中 `status = 'pending'` 的行；
2. 通过 NFS 挂载访问 NAS 中的 `img_path`；
3. 调用 OmniParser (YOLOv8 + LayoutLMv3) 抽取 UI 文字；
4. 将 `json_payload` 与摘要写回数据库；
5. 若失败，`status` 标记 `error` 并记录日志。
6. 現已取消批次處理，任務依序解析。
7. 若需跳過 SHA-256 驗證，可在啟動時加入 `--skip-sha256-check`。

### 4. RouterOS 心跳监控 (`hb_inotify.py`)

* 透过 inotify 监视截图目录新增，配合 ARP 与 DHCP 清单更新防火墙名单。
* 周期性比对未上传截图的 IP，并自动加入阻断列表。

### 5. Web 前端（草稿）

* **后端**：FastAPI，运行在 NAS Docker；提供图像预览、OCR JSON 展示、全文搜索与权限管理 API。
* **前端**：React + Tailwind + shadcn/ui，可视化图表使用 Recharts。
* **连接**：通过环境变量 `DATABASE_URL` 直连 GPU Workstation 上的 Postgres。
* **启动草稿**：`uvicorn webui.app.main:app --reload` ，浏览器打开 `http://localhost:8000/static/index.html` 查看介面。
* **单张测试**：`python single_image_gradio.py`，以 Gradio 介面上传图片解析。

## 数据库概要

| 表名            | 说明                                           |
| ------------- | -------------------------------------------- |
| captures      | 主键 id, img\_path, json\_payload, created\_at |
| captures\_tmp | staging，用作 `COPY`                            |
| workers\_log  | 解析进程运行记录                                     |

> 默认凭证：`postgres://omniapp:<password>@192.168.1.240:5432/omni`，请通过环境变量 `DATABASE_URL` 注入。

## 部署步骤速览

```bash
# Synology NAS
# 1. 启动 FTP + 创建 /volume1/screens
# 2. (可选) 部署 Web‑UI
#    uvicorn webui.app.main:app --reload

# GPU Workstation
sudo apt update && sudo apt install -y python3-venv build-essential postgresql-16 postgresql-contrib
# PostgreSQL 初始化
sudo -u postgres createuser omniapp --pwprompt
sudo -u postgres createdb -O omniapp omni

# Python 环境
python3 -m venv venv && . venv/bin/activate
pip install -r OmniParser/requirements.txt
python worker_batch.py
```

## 维护与监控

| 任务                            | 频率     | 工具/位置              |
| ----------------------------- | ------ | ------------------ |
| NAS 磁盘使用量检查                   | Daily  | DSM > 存储管理         |
| PostgreSQL `VACUUM / ANALYZE` | Weekly | `pg_cron`          |
| GPU 温度与显存监控                   | Live   | `watch nvidia-smi` |
| worker & sync 日志轮转            | Weekly | `logrotate.d/omni` |
| Web‑UI 容器健康检查 (WIP)           | TBD    | Docker Healthcheck |

## Roadmap

* [ ] **Web 前端**：OCR 结果浏览、全文检索、告警订阅。
* [ ] 引入 OpenTelemetry 以串接全链路 Trace。
* [ ] 将 FTP 上传改为 HTTPS + JWT 验证。
* [ ] 自动化 Helm Chart 以便后续迁移到 K8s。
