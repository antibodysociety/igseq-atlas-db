"""INN Audit Log — opt-in per-INN change tracking across the motif pipeline.

Classes
-------
InnAuditLog     Append-only audit trail for INN set changes.

Functions
---------
start_inn_audit   Activate the audit log.
get_inn_audit     Return the currently active log.
stop_inn_audit    Stop the log and return it.
_audit            Write one entry (no-op if inactive).
_audit_set        Write entries for every INN in a set (no-op if inactive).
"""

from __future__ import annotations

from threading import Lock as _Lock
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# INN Audit Log
# ---------------------------------------------------------------------------


class InnAuditLog:
    """Append-only audit trail for INN set changes throughout the motif pipeline.

    Each time an INN is added to, removed from, or moved between a motif's
    ``inns`` or ``adjusted_inns`` set, a structured entry is appended.

    Usage::

        from inn_analysis_ops.analytics.co_mutation_analytics import start_inn_audit, stop_inn_audit
        start_inn_audit()                       # track all INNs
        start_inn_audit({'faricimab', 'xyz'})   # track subset
        result = build_constant_chain_analysis(ds, cfg)
        log = stop_inn_audit()
        print(log.to_dataframe().query("inn == 'faricimab'"))
    """

    def __init__(self, track: Optional[set[str]] = None) -> None:
        self._track = track  # None = track all INNs
        self._log: dict[str, list[dict]] = {}
        self._counter = 0
        self._lock = _Lock()

    def record(
        self,
        inn: str,
        step: str,
        motif: str,
        action: str,
        reason: str = "",
    ) -> None:
        """Append one entry for *inn*."""
        if self._track is not None and inn not in self._track:
            return
        with self._lock:
            self._counter += 1
            self._log.setdefault(inn, []).append(
                {
                    "step_idx": self._counter,
                    "step": step,
                    "motif": motif,
                    "action": action,
                    "reason": reason,
                }
            )

    def record_set(
        self,
        inns,
        step: str,
        motif: str,
        action: str,
        reason: str = "",
    ) -> None:
        """Append one entry per INN in *inns*."""
        if _ACTIVE_AUDIT is None:
            return
        for inn in inns:
            self.record(inn, step, motif, action, reason)

    def get(self, inn: str) -> list[dict]:
        """Return all entries for *inn*, sorted by step_idx."""
        return sorted(self._log.get(inn, []), key=lambda e: e["step_idx"])

    def to_dataframe(self) -> pd.DataFrame:
        """Return all entries as a flat DataFrame sorted by inn and step_idx."""
        rows = [
            {"inn": inn, **e}
            for inn, entries in self._log.items()
            for e in entries
        ]
        if not rows:
            return pd.DataFrame(
                columns=[
                    "inn",
                    "step_idx",
                    "step",
                    "motif",
                    "action",
                    "reason",
                ]
            )
        return (
            pd.DataFrame(rows)
            .sort_values(["inn", "step_idx"])
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return sum(len(v) for v in self._log.values())


_ACTIVE_AUDIT: Optional[InnAuditLog] = None


def start_inn_audit(track: Optional[set[str]] = None) -> InnAuditLog:
    """Activate the INN audit log.

    Parameters
    ----------
    track : set[str], optional
        INNs to track.  ``None`` (default) tracks every INN.
    """
    global _ACTIVE_AUDIT
    _ACTIVE_AUDIT = InnAuditLog(track=track)
    return _ACTIVE_AUDIT


def get_inn_audit() -> Optional[InnAuditLog]:
    """Return the currently active audit log, or ``None`` if not started."""
    return _ACTIVE_AUDIT


def stop_inn_audit() -> Optional[InnAuditLog]:
    """Stop the audit log and return the completed log object."""
    global _ACTIVE_AUDIT
    log = _ACTIVE_AUDIT
    _ACTIVE_AUDIT = None
    return log


def _audit(
    inn: str, step: str, motif: str, action: str, reason: str = ""
) -> None:
    """Write one audit entry. No-op if audit is not active."""
    if _ACTIVE_AUDIT is not None:
        _ACTIVE_AUDIT.record(inn, step, motif, action, reason)


def _audit_set(
    inns, step: str, motif: str, action: str, reason: str = ""
) -> None:
    """Write audit entries for every INN in *inns*. No-op if audit inactive."""
    if _ACTIVE_AUDIT is not None:
        _ACTIVE_AUDIT.record_set(inns, step, motif, action, reason)
