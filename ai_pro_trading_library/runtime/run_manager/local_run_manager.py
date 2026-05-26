"""Local subprocess lifecycle manager — port of standard_bot LocalRunManager.

Runs a daemon command (e.g. `python -m ai_pro_trading_library.runtime.daemon
--spec ...`) as a detached background process, persists pid + status to
disk, and exposes refresh / stop / kill primitives. Status is read from
disk so a separate process can inspect a running daemon.

Field-set parity vs `cyqnt_trd.standard_bot.runtime.run_manager.ManagedRunRecord`.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class ManagedRunRecord:
    """Subprocess lifecycle record. Field set mirrors standard_bot."""

    run_id: str
    profile: str
    command: list[str]
    pid: int
    pgid: int | None
    status: str  # "running" / "stopping" / "stopped" / "killing" / "killed" / "exited"
    started_at: int
    updated_at: int
    workdir: str
    log_path: str
    metadata_path: str
    stop_requested_at: int | None = None
    ended_at: int | None = None
    tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ManagedRunRecord":
        return cls(
            run_id=str(payload["run_id"]),
            profile=str(payload["profile"]),
            command=list(payload.get("command", [])),
            pid=int(payload["pid"]),
            pgid=None if payload.get("pgid") is None else int(payload["pgid"]),
            status=str(payload["status"]),
            started_at=int(payload["started_at"]),
            updated_at=int(payload["updated_at"]),
            workdir=str(payload["workdir"]),
            log_path=str(payload["log_path"]),
            metadata_path=str(payload["metadata_path"]),
            stop_requested_at=None
            if payload.get("stop_requested_at") is None
            else int(payload["stop_requested_at"]),
            ended_at=None if payload.get("ended_at") is None else int(payload["ended_at"]),
            tags=dict(payload.get("tags", {})),
        )


class LocalRunManager:
    """Subprocess pid/status/kill manager. Mirrors standard_bot LocalRunManager."""

    def __init__(self, root_dir: str | Path = ".ai_pro_trading_runs") -> None:
        self.root_dir = Path(root_dir)
        self.records_dir = self.root_dir / "records"
        self.logs_dir = self.root_dir / "logs"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def start_command(
        self,
        *,
        profile: str,
        command: Sequence[str],
        workdir: str,
        run_id: str | None = None,
        log_path: str | None = None,
        env: dict[str, str] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> ManagedRunRecord:
        resolved_run_id = run_id or str(uuid.uuid4())
        metadata_path = self.records_dir / f"{resolved_run_id}.json"
        if metadata_path.exists():
            raise FileExistsError(f"run_id already exists: {resolved_run_id}")

        resolved_log_path = (
            Path(log_path) if log_path else self.logs_dir / f"{resolved_run_id}.log"
        )
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

        runtime_env = os.environ.copy()
        runtime_env.setdefault("AI_PRO_TRADING_MANAGED_RUN_ID", resolved_run_id)
        runtime_env.setdefault("AI_PRO_TRADING_MANAGED_PROFILE", profile)
        if env:
            runtime_env.update(env)

        with resolved_log_path.open("ab") as sink:
            process = subprocess.Popen(
                list(command),
                cwd=workdir,
                env=runtime_env,
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        try:
            pgid = os.getpgid(process.pid)
        except OSError:
            pgid = None

        record = ManagedRunRecord(
            run_id=resolved_run_id,
            profile=profile,
            command=list(command),
            pid=process.pid,
            pgid=pgid,
            status="running",
            started_at=_now_ms(),
            updated_at=_now_ms(),
            workdir=str(workdir),
            log_path=str(resolved_log_path),
            metadata_path=str(metadata_path),
            tags=dict(tags or {}),
        )
        self._write_record(record)
        return record

    def list_runs(self, *, refresh: bool = True) -> list[ManagedRunRecord]:
        records: list[ManagedRunRecord] = []
        for path in sorted(self.records_dir.glob("*.json")):
            rec = self._read_record(path)
            if refresh:
                rec = self.refresh(rec.run_id)
            records.append(rec)
        return sorted(records, key=lambda item: item.started_at, reverse=True)

    def get_run(self, run_id: str, *, refresh: bool = True) -> ManagedRunRecord:
        path = self.records_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"unknown run_id: {run_id}")
        rec = self._read_record(path)
        return self.refresh(run_id) if refresh else rec

    def refresh(self, run_id: str) -> ManagedRunRecord:
        rec = self.get_run(run_id, refresh=False)
        alive = self._is_alive(rec.pid)
        updated = False
        if alive and rec.status not in {"running", "stopping"}:
            rec.status = "running"
            rec.ended_at = None
            updated = True
        elif not alive and rec.status not in {"exited", "stopped", "killed"}:
            rec.status = "exited"
            rec.ended_at = _now_ms()
            updated = True
        if updated:
            rec.updated_at = _now_ms()
            self._write_record(rec)
        return rec

    def stop(self, run_id: str, *, timeout_seconds: float = 5.0) -> ManagedRunRecord:
        return self._terminate(run_id, sig=signal.SIGTERM, timeout_seconds=timeout_seconds)

    def kill(self, run_id: str, *, timeout_seconds: float = 2.0) -> ManagedRunRecord:
        return self._terminate(run_id, sig=signal.SIGKILL, timeout_seconds=timeout_seconds)

    def _terminate(
        self,
        run_id: str,
        *,
        sig: signal.Signals,
        timeout_seconds: float,
    ) -> ManagedRunRecord:
        rec = self.get_run(run_id, refresh=False)
        if not self._is_alive(rec.pid):
            if rec.status not in {"exited", "stopped", "killed"}:
                rec.status = "exited"
                rec.ended_at = _now_ms()
                rec.updated_at = _now_ms()
                self._write_record(rec)
            return rec

        rec.stop_requested_at = _now_ms()
        rec.status = "stopping" if sig == signal.SIGTERM else "killing"
        rec.updated_at = _now_ms()
        self._write_record(rec)

        target = -rec.pgid if rec.pgid else rec.pid
        try:
            os.kill(target, sig)
        except ProcessLookupError:
            rec.status = "exited"
            rec.ended_at = _now_ms()
            rec.updated_at = _now_ms()
            self._write_record(rec)
            return rec

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not self._is_alive(rec.pid):
                break
            time.sleep(0.1)

        if self._is_alive(rec.pid):
            rec.status = "running"
        else:
            rec.status = "stopped" if sig == signal.SIGTERM else "killed"
            rec.ended_at = _now_ms()
        rec.updated_at = _now_ms()
        self._write_record(rec)
        return rec

    def _read_record(self, path: Path) -> ManagedRunRecord:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ManagedRunRecord.from_dict(payload)

    def _write_record(self, record: ManagedRunRecord) -> None:
        path = Path(record.metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _is_alive(self, pid: int) -> bool:
        # First reap zombies if we're the parent
        try:
            waited_pid, _ = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                return False
        except (ChildProcessError, OSError):
            pass
        # Probe via signal 0
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        # Filter out zombies via ps state
        ps = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
        state = ps.stdout.strip()
        if not state:
            return False
        if "Z" in state:
            return False
        return True


__all__ = ["LocalRunManager", "ManagedRunRecord"]
