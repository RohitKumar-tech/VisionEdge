# VisionEdge

AI-powered IP camera SaaS platform. Face recognition, staff attendance, VIP detection, and blacklist alerts — all running on-site with a private NAS backend.

---

## Repository structure

```
VisionEdge/
├── edge-agent/          # Person A — runs on Mini PC / Jetson at client site
├── backend/             # Person B — FastAPI on Synology NAS (Docker)
├── web/                 # Person B — Next.js dashboard
├── mobile/              # Person B — React Native client app
├── infra/               # Docker Compose, nginx config, deployment scripts
├── docs/                # Architecture, decisions, specs
├── API_CONTRACT.md      # Source of truth for edge↔backend communication
└── VisionEdge_Stakeholder.pptx
```

---

## System overview

```
[Client Site]
  IP Cameras (RTSP)
      → Edge Mini PC (face detection + recognition)
      → pushes events + RTMP stream → NAS

[Synology NAS — your central backend]
  FastAPI + PostgreSQL + mediamtx (video relay)
  Accessible via Cloudflare Tunnel at api.visionedge.in

[Client App — anywhere in the world]
  Live camera feeds (HLS over HTTPS)
  Attendance dashboard, alerts, AI features

[AWS — minimum]
  Firebase FCM (push notifications)
  SES (email alerts)
  S3 (nightly encrypted DB backup)
  Route53 (DNS)
```

Full architecture: [`docs/architecture.md`](docs/architecture.md)

---

## Edge Agent

Runs on a Mini PC or Jetson Nano at the client site. Requires no internet — queues events locally and syncs when online.

### What it does
- Connects to all RTSP cameras (configurable per site)
- Detects faces using InsightFace buffalo_l (ResNet100 + ArcFace)
- Matches against enrolled persons from local SQLite DB
- Fires typed events: `checkin`, `checkout`, `vip_spotted`, `blacklist_alert`, `unknown_face`
- Anti-spoof: rejects printed photos and phone screens (liveness score + 3-frame confirmation)
- Attendance: one check-in per staff per day, checkout tracked with latest exit time
- Queues all events in SQLite when offline — uploads when connection restored
- Syncs face DB from backend every 60s

### Setup

```bash
cd edge-agent
python -m venv .venv
source .venv/bin/activate   # fish: source .venv/bin/activate.fish
pip install -r requirements.txt
cp .env.example .env        # fill in SITE_ID, AGENT_TOKEN, CLOUD_BASE_URL, cameras
python main.py
```

### Camera config (`.env`)

```env
SITE_ID=site-uuid
AGENT_TOKEN=your-token
CLOUD_BASE_URL=https://api.visionedge.in

# Cameras as JSON
CAMERAS=[
  {"camera_id": "cam-1", "rtsp_url": "rtsp://admin:pass@192.168.1.10/stream1", "role": "entry"},
  {"camera_id": "cam-2", "rtsp_url": "rtsp://admin:pass@192.168.1.11/stream1", "role": "exit"},
  {"camera_id": "cam-3", "rtsp_url": "rtsp://admin:pass@192.168.1.12/stream1", "role": "both"}
]
```

Camera roles:
- `entry` — fires `checkin` events
- `exit` — fires `checkout` events
- `both` — fires `checkin` only (single-door shops)

### Face recognition model

Uses InsightFace `buffalo_l` (no GPU required — runs on CPU via ONNX Runtime).
Fine-tuned ArcFace head on IMFDB dataset (100 Indian celebrity identities) for improved accuracy on Indian faces.

Training pipeline: [`edge-agent/training/`](edge-agent/training/README.md)

---

## Development status

| Component | Status | Owner |
|-----------|--------|-------|
| Edge Agent core | ✅ Complete | Person A |
| Multi-camera parallel processing | ✅ Complete | Person A |
| Attendance (check-in/out) | ✅ Complete | Person A |
| Anti-spoof (2-layer) | ✅ Complete | Person A |
| Face sync from backend | ✅ Complete | Person A |
| Event upload (offline queue) | ✅ Complete | Person A |
| Edge RTMP push (live stream) | 🔲 Pending | Person A |
| NAS Docker stack | 🔲 Pending | Person B |
| FastAPI backend | 🔲 Pending | Person B |
| PostgreSQL schema | 🔲 Pending | Person B |
| Dashboard (web) | 🔲 Pending | Person B |
| Mobile app | 🔲 Pending | Person B |
| Push notifications (FCM) | 🔲 Pending | Person B |
| AWS integrations (SES, S3) | 🔲 Pending | Person B |
| Deployment (systemd, Docker) | 🔲 Pending | Both |

---

## API contract

[`API_CONTRACT.md`](API_CONTRACT.md) — agreed interface between edge agent and backend. **Do not change without coordinating both sides.**
