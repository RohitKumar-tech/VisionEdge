# VisionEdge — Architecture

## System diagram

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLIENT SITE  (one per shop / office)
┌─────────────────────────────────────┐
│  IP Cameras (RTSP, e.g. Tapo C200)  │
│       ↓                             │
│  Edge Mini PC / Jetson Nano         │
│  ├─ AI: detect + recognise faces    │
│  ├─ Events → POST /api/v1/events    │
│  └─ Video → RTMP push to NAS ───────┼──→ internet
└─────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYNOLOGY NAS  (your central backend, 8TB)
┌──────────────────────────────────────────────────────┐
│  Docker containers:                                   │
│                                                       │
│  ┌───────────────┐   ┌─────────────────────────────┐ │
│  │  mediamtx     │   │  FastAPI Backend             │ │
│  │  RTMP → HLS   │   │  + JWT auth                 │ │
│  │  (video relay)│   │  + REST API                 │ │
│  └──────┬────────┘   └────────────┬────────────────┘ │
│         │                         │                   │
│  ┌──────▼─────────────────────────▼───────┐          │
│  │           PostgreSQL                    │          │
│  │  sites / persons / events / attendance  │          │
│  └─────────────────────────────────────────┘         │
│                                                       │
│  Accessible via Cloudflare Tunnel                     │
│  → api.visionedge.in  (no port forwarding needed)     │
└──────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLIENT APP  (iOS / Android / Web — anywhere in world)
┌──────────────────────────────────┐
│  ├─ Login → JWT (scoped to site) │
│  ├─ Live camera feed (HLS) ◄─────┼── HTTPS from NAS
│  ├─ Attendance dashboard    ◄────┼── REST API
│  ├─ Real-time alerts             │
│  └─ AI features                  │
└──────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AWS  (minimum — only what NAS can't do)
  Firebase FCM  → push notifications to mobile app
  SES           → transactional email (alerts, reports)
  S3            → nightly encrypted DB backup from NAS
  Route53       → DNS management for visionedge.in
  Estimated cost: ~$5–15/month
```

---

## On-demand video streaming

Video is NOT streamed continuously. Only when a client actively views a camera.

```
1. Client taps camera in app
2. App: GET /api/v1/stream-token?camera_id=cam-2&site_id=abc
3. NAS returns: { hls_url, token (15min expiry) }
4. NAS signals edge agent: "start pushing cam-2"
5. Edge: ffmpeg RTSP → RTMP → NAS mediamtx
6. mediamtx: RTMP → HLS segments
7. App plays HLS with token in Authorization header
8. Client closes feed → NAS tells edge to stop pushing
```

This saves bandwidth — no video sent when nobody is watching.

---

## Security model

| Layer | Mechanism |
|-------|-----------|
| Edge → NAS | HTTPS + per-site `agent_token` (Bearer) |
| App → NAS | HTTPS + user JWT (RS256, 24h expiry) |
| Video streams | Short-lived stream tokens (15 min, scoped to `camera_id + site_id`) |
| Multi-tenancy | JWT contains `site_id` — all queries filtered by it |
| Face embeddings | Never leave NAS — app only receives names and event metadata |
| NAS to internet | Cloudflare Tunnel — your IP never exposed |
| DB backups | AES-256 encrypted before S3 upload |

---

## Edge Agent internals

```
main.py
  ├── StreamManager          one thread per camera, RTSP capture + auto-reconnect
  ├── FaceDetector           InsightFace buffalo_l, CLAHE enhancement, liveness score
  ├── Recognizer             cosine similarity matching, role→event_type mapping
  ├── recognition_loop       ThreadPoolExecutor (one worker per camera)
  │     ├── anti-spoof       Layer 1: Laplacian variance (≥50 = live face)
  │     │                    Layer 2: 3-frame confirmation in 10s window
  │     ├── checkin logic    once per day via attendance_log (survives restarts)
  │     └── checkout logic   30min cooldown, always updates latest exit time
  ├── FaceSync               polls GET /api/v1/faces/sync every 60s
  └── EventUploader          SQLite offline queue, drains every 5s, exp. backoff
```

### Camera roles

| Role | Behaviour |
|------|-----------|
| `entry` | Fires `checkin` on first recognition of day |
| `exit` | Fires `checkout` (30min cooldown, updates latest exit time) |
| `both` | Fires `checkin` only — for single-door setups |

### Anti-spoof (staff only)

Two layers required before attendance is logged:

1. **Liveness score** — Laplacian variance of face crop. Real faces ≥ 50, photos/screens < 50.
2. **Multi-frame confirmation** — person must be detected 3× within 10s. Defeats quickly-flashed phone photo.

---

## NAS Docker stack (Person B)

```yaml
services:
  postgres:    image: postgres:16
  backend:     image: visionedge/backend   # FastAPI
  dashboard:   image: visionedge/web       # Next.js
  mediamtx:    image: bluenviron/mediamtx  # RTMP → HLS relay
  nginx:       image: nginx:alpine         # SSL termination + reverse proxy
```

Cloudflare Tunnel agent also runs on NAS, routing `api.visionedge.in` → nginx.

---

## Data residency

| Data | Where stored | Rationale |
|------|-------------|-----------|
| Face embeddings | NAS PostgreSQL | Biometric data — never on public cloud |
| Attendance records | NAS PostgreSQL | Client operational data |
| Recognition events | NAS PostgreSQL | Client operational data |
| Video clips | NAS filesystem (8TB) | Large files, on-prem |
| DB backups | AWS S3 (encrypted) | Offsite disaster recovery only |
| Push notification tokens | AWS (Firebase) | Required by FCM |

---

## Key decisions log

| Decision | Rationale |
|----------|-----------|
| Synology NAS as primary backend | Privacy (biometrics on-prem), cost (no cloud DB fees), 8TB storage |
| Cloudflare Tunnel over port forwarding | Hides NAS IP, free, automatic HTTPS |
| On-demand streaming | Saves NAS upload bandwidth |
| SQLite on edge | Works fully offline, zero config |
| InsightFace buffalo_l | Production-grade, runs on CPU, no GPU needed at edge |
| Per-site agent tokens | Compromise of one site doesn't expose other clients |
| attendance_log in edge SQLite | Survives restart — no duplicate check-in events |
