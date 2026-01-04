import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.models import AuditLog, User

def write_event(
    db: Session,
    case_id: str,
    actor: User,
    action: str,
    payload: Dict[str, Any],
    file_hash: Optional[str] = None,
):
    """
    Writes an immutable, hash-chained event to the audit log.

    The hash is computed as:
    SHA256(prev_hash || canonical_json(payload) || file_hash || created_at_isoformat)
    """
    # 1. Get the hash of the previous log entry for this case
    # Use insertion order (id) to determine the previous entry reliably.
    last_entry = (
        db.query(AuditLog)
        .filter(AuditLog.case_id == case_id)
        .order_by(desc(AuditLog.id))
        .first()
    )
    prev_hash = last_entry.hash if last_entry else "genesis"

    # 2. Prepare data for hashing
    # Normalize timestamp to second precision to avoid storage/runtime
    # representation differences (SQLite may truncate microseconds).
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    # Use separators to prevent ambiguity, and sort keys for canonical form
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    
    # 3. Create the new log entry without the final hash first so that the DB can
    # assign a canonical `created_at` value (avoids mismatch between Python and DB timestamps).
    new_log_entry = AuditLog(
        case_id=case_id,
        actor_id=actor.id,
        action=action,
        payload_json=canonical_payload,
        file_hash=file_hash,
        prev_hash=prev_hash,
        hash="",  # placeholder
    )
    db.add(new_log_entry)
    db.commit()
    db.refresh(new_log_entry)

    # 4. Now compute the hash using the actual stored `created_at` from the DB
    actual_created_at = new_log_entry.created_at
    hash_content = (
        f"{new_log_entry.prev_hash}{new_log_entry.payload_json}{new_log_entry.file_hash or ''}{actual_created_at.isoformat()}"
    ).encode("utf-8")
    current_hash = hashlib.sha256(hash_content).hexdigest()

    # 5. Update the entry with the computed hash and persist
    new_log_entry.hash = current_hash
    db.add(new_log_entry)
    db.commit()
    db.refresh(new_log_entry)
    return new_log_entry
