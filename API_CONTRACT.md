# VisionEdge — API Contract

> **Source of truth for edge agent ↔ backend communication.**
> Do not change without updating both Person A (edge-agent) and Person B (backend).
> Last updated: 2026-04-17

---

## Authentication

Two token types:

| Token | Used by | Header | Scope |
|-------|---------|--------|-------|
| `agent_token` | Edge Agent | `Authorization: Bearer <token>` | One token per site — all cameras at that site |
| `user_jwt` | Mobile / Web App | `Authorization: Bearer <token>` | Scoped to `site_id`, 24h expiry |
| `stream_token` | App video player | `Authorization: Bearer <token>` | Scoped to `camera_id + site_id`, 15min expiry |

---

## Edge Agent → Backend

### POST /api/v1/events

Sent by edge agent for every recognition event.

**Auth:** `agent_token`

**Request:**
```json
{
  "camera_id":   "cam-uuid",
  "person_id":   "person-uuid or null",
  "person_name": "Alice Chen or null",
  "role":        "staff | vip | blacklisted | null",
  "confidence":  0.94,
  "event_type":  "checkin | checkout | vip_spotted | blacklist_alert | unknown_face",
  "timestamp":   "2026-04-17T10:32:01Z",
  "frame_path":  null
}
```

**Field rules:**

| Field | Required | Notes |
|-------|----------|-------|
| `camera_id` | yes | Must belong to this agent's site |
| `person_id` | no | `null` for unknown faces |
| `person_name` | no | Denormalised — source of truth is persons table |
| `role` | no | `null` for unknown. Values: `staff`, `vip`, `blacklisted` |
| `confidence` | yes | Float 0–1. Edge only sends ≥ confidence_threshold |
| `event_type` | yes | Full set required — backend routes on this value |
| `timestamp` | yes | ISO 8601 UTC, set at moment of detection on edge |
| `frame_path` | no | Local path on edge device (backend may request clip later) |

**event_type values:**
```
checkin         — staff arrived (first recognition today, anti-spoof passed)
checkout        — staff left (exit camera, 30min cooldown)
vip_spotted     — VIP detected (priority alert)
blacklist_alert — blacklisted person detected (security alert)
unknown_face    — face detected but not matched
```

**Response:**
```json
201 Created
{ "event_id": "server-uuid" }
```

**Errors:**
| Code | Reason |
|------|--------|
| 401 | Invalid or missing agent_token |
| 403 | camera_id not registered under this agent's site |
| 422 | Missing required field or invalid event_type |

---

### GET /api/v1/faces/sync

Edge agent calls this every 60s to get updated face database.

**Auth:** `agent_token`

**Query params:**

| Param | Required | Example |
|-------|----------|---------|
| `since` | no | `2026-04-17T10:00:00Z` — only changes after this time |

**Response:**
```json
200 OK
{
  "actions": [
    {
      "action":     "add",
      "id":         "person-uuid",
      "name":       "Alice Chen",
      "type":       "staff | vip | blacklisted",
      "embedding":  "<base64-encoded float32 bytes>",
      "updated_at": "2026-04-17T09:00:00Z"
    },
    {
      "action": "delete",
      "id":     "person-uuid"
    }
  ],
  "synced_at": "2026-04-17T10:35:00Z"
}
```

> **Note for Person A:** `embedding` is base64-encoded raw float32 bytes (512 floats = 2048 bytes).
> Decode with: `np.array(struct.unpack("512f", base64.b64decode(b64str)), dtype=np.float32)`

> **Note for Person B:** Do NOT send plain JSON float array — base64 is ~3× smaller and avoids float precision loss.

---

## App → Backend (User-facing API)

### POST /api/v1/auth/login

**Request:**
```json
{ "email": "client@shop.com", "password": "..." }
```

**Response:**
```json
{
  "access_token": "<jwt>",
  "site_id": "site-uuid",
  "expires_in": 86400
}
```

---

### GET /api/v1/attendance

**Auth:** `user_jwt`

**Query params:**

| Param | Example | Notes |
|-------|---------|-------|
| `date` | `2026-04-17` | UTC date. Defaults to today |
| `person_id` | optional | Filter to one person |

**Response:**
```json
{
  "date": "2026-04-17",
  "records": [
    {
      "person_id":      "uuid",
      "name":           "Rahul Sharma",
      "checked_in_at":  "2026-04-17T03:32:01Z",
      "checked_out_at": "2026-04-17T13:05:42Z",
      "first_camera":   "cam-1"
    }
  ]
}
```

---

### GET /api/v1/events

**Auth:** `user_jwt`

**Query params:**

| Param | Example | Notes |
|-------|---------|-------|
| `since` | `2026-04-17T00:00:00Z` | |
| `event_type` | `blacklist_alert` | optional filter |
| `limit` | `50` | default 50, max 200 |

**Response:**
```json
{
  "events": [
    {
      "event_id":    "uuid",
      "camera_id":   "cam-1",
      "person_name": "Alice",
      "event_type":  "checkin",
      "confidence":  0.97,
      "timestamp":   "2026-04-17T03:32:01Z"
    }
  ]
}
```

---

### GET /api/v1/stream-token

Returns a short-lived token to play a camera's live HLS stream.

**Auth:** `user_jwt`

**Query params:**

| Param | Required | Notes |
|-------|----------|-------|
| `camera_id` | yes | Must belong to user's site |

**Response:**
```json
{
  "hls_url":    "https://api.visionedge.in/live/{site_id}/{camera_id}/stream.m3u8",
  "token":      "<stream_token>",
  "expires_in": 900
}
```

> **Note for Person B:** After issuing this token, signal the edge agent (via WebSocket or polling) to start pushing RTMP for this camera. Stop pushing when token expires and no refresh is requested.

---

### GET /api/v1/persons

**Auth:** `user_jwt`

**Response:**
```json
{
  "persons": [
    {
      "id":         "uuid",
      "name":       "Alice Chen",
      "type":       "staff | vip | blacklisted",
      "updated_at": "2026-04-17T00:00:00Z"
    }
  ]
}
```

---

### POST /api/v1/persons

Enrol a new person.

**Auth:** `user_jwt`

**Request:** `multipart/form-data`
```
name:  "Alice Chen"
type:  "staff"
photos: [file1.jpg, file2.jpg, file3.jpg]   ← 3–5 photos recommended
```

**Response:**
```json
201 Created
{ "person_id": "uuid" }
```

> **Note for Person B:** Backend must extract embedding using buffalo_l, average across all photos, store normalised float32 blob. Next face sync will push this to all edge agents at this site.

---

### DELETE /api/v1/persons/{person_id}

**Auth:** `user_jwt`

**Response:**
```json
200 OK
{ "deleted": true }
```

> Next face sync will send a `delete` action to all edge agents.

---

## Changes log

| Date | Change | Agreed by |
|------|--------|-----------|
| 2026-04-12 | Initial contract (events + faces/sync) | Person A + B |
| 2026-04-17 | Added checkout event_type | Person A |
| 2026-04-17 | Added stream-token endpoint | Person A + B |
| 2026-04-17 | Added attendance, events, persons endpoints | Person A |
| 2026-04-17 | Changed embedding format: JSON array → base64 float32 | Person A |
| 2026-04-17 | Added multi-tenancy security model | Person A |
