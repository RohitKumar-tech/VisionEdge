# VisionEdge — API Contract

> **LOCKED. Do not change mid-week without updating both Person A (edge-agent) and Person B (backend) simultaneously.**
> Last agreed: 2026-04-12

---

## POST /api/v1/events

**Auth:** `Authorization: Bearer <agent_token>` (Edge Agent token, not user JWT)

**Sent by:** Edge Agent → Cloud Backend on every face detection event.

### Request body

```json
{
  "camera_id":   "cam-uuid",
  "person_id":   "person-uuid or null",
  "person_name": "Alice Chen or null",
  "role":        "staff | vip | blacklisted | null",
  "confidence":  0.94,
  "event_type":  "checkin | checkout | vip_spotted | blacklist_alert | unknown_face",
  "timestamp":   "2026-04-12T10:32:01Z",
  "frame_path":  "optional — local absolute path to saved JPEG on edge device"
}
```

### Field rules

| Field | Required | Notes |
|---|---|---|
| `camera_id` | yes | Must match a camera registered under this agent's site |
| `person_id` | no | `null` when face detected but not matched (unknown_face) |
| `person_name` | no | `null` when unknown. Denormalized for fast dashboard display — source of truth is persons table |
| `role` | no | `null` when unknown. Values: `staff`, `vip`, `blacklisted` |
| `confidence` | yes | Float 0–1. Edge agent only sends events with confidence ≥ 0.95 |
| `event_type` | yes | See valid values below |
| `timestamp` | yes | ISO 8601 UTC. Set by Edge Agent at moment of detection |
| `frame_path` | no | Local path on edge device. Backend may request frame upload separately |

### Valid event_type values

```
checkin          — known person arrived (first detection of day / after absence)
checkout         — known person left (inferred from absence after checkin)
vip_spotted      — VIP person detected (triggers priority alert)
blacklist_alert  — blacklisted person detected (triggers security alert)
unknown_face     — face detected but not matched to any known person
```

> **Note for Person B:** `event_type` is NOT simplified to `known|unknown`. The full set is required because backend routing logic (Celery workers, notification priority, attendance engine) depends on the specific type. The backend's `recognition_events` table and the edge agent's SQLite `pending_events` table both use these exact values.

### Response

```json
201 Created
{ "event_id": "server-assigned-uuid" }
```

### Error responses

| Code | Reason |
|---|---|
| 401 | Invalid or missing agent token |
| 403 | camera_id does not belong to this agent's site/client |
| 422 | Missing required field or invalid event_type value |

---

## GET /api/v1/sync/faces

**Auth:** `Authorization: Bearer <agent_token>`

**Called by:** Edge Agent every 60 seconds to get updated face database.

### Query params

| Param | Required | Example |
|---|---|---|
| `since` | no | `2026-04-12T10:00:00Z` — only return changes after this time |

### Response

```json
200 OK
{
  "actions": [
    {
      "action":     "add",
      "person_id":  "uuid",
      "name":       "Alice Chen",
      "type":       "staff | vip | blacklisted",
      "embedding":  [0.12, -0.34, ...]
    },
    {
      "action":    "delete",
      "person_id": "uuid"
    }
  ],
  "synced_at": "2026-04-12T10:35:00Z"
}
```

> **Note for Person A:** `embedding` is a plain JSON array of 512 floats (decrypted by backend before sending). Store as `BLOB` in SQLite using `np.array(embedding, dtype=np.float32).tobytes()`.

---

## Changes log

| Date | Change | Agreed by |
|---|---|---|
| 2026-04-12 | Initial contract locked | Person A + Person B |
