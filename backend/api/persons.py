"""Staff / VIP / blacklist management, face enrollment."""
# TODO Phase 2: Implement
from fastapi import APIRouter

router = APIRouter(prefix="/persons", tags=["persons"])

# GET    /persons              → list (filter by type)
# POST   /persons              → create person + enroll face photo
# PUT    /persons/{id}         → update details
# DELETE /persons/{id}         → permanent delete (DPDP compliance) — removes person + all events + clips
# POST   /persons/{id}/photo   → re-enroll face photo
