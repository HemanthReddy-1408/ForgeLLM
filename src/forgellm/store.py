"""Experiment + registry store. Postgres when `FORGELLM_POSTGRES_DSN` is set and reachable (tables live in their own
`forgellm` schema, so a shared database is not touched); otherwise a local SQLite file. Same API either way.

Tables: runs, metrics, adapters (the model registry), eval_reports."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from forgellm.config import ARTIFACTS, ROOT


def _load_env() -> None:
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS {s}runs (id TEXT PRIMARY KEY, kind TEXT, name TEXT, config TEXT, status TEXT,
       summary TEXT, started_at DOUBLE PRECISION, finished_at DOUBLE PRECISION)""",
    """CREATE TABLE IF NOT EXISTS {s}metrics (run_id TEXT, step INTEGER, payload TEXT)""",
    """CREATE TABLE IF NOT EXISTS {s}adapters (name TEXT, version INTEGER, base_model TEXT, method TEXT, task TEXT,
       path TEXT, status TEXT, num_parameters BIGINT, data_hash TEXT, config_hash TEXT, metrics TEXT, gate TEXT,
       created_at DOUBLE PRECISION, PRIMARY KEY (name, version))""",
    """CREATE TABLE IF NOT EXISTS {s}eval_reports (id TEXT PRIMARY KEY, model_tag TEXT, suite TEXT, report TEXT, created_at DOUBLE PRECISION)""",
]


class Store:
    def __init__(self, dsn: str | None = None, sqlite_path: str | Path | None = None, schema: str = "forgellm") -> None:
        _load_env()
        self.backend = "sqlite"
        self.error: str | None = None
        dsn = os.environ.get("FORGELLM_POSTGRES_DSN") if dsn is None else dsn   # dsn="" means "no Postgres" (tests, offline)
        self.conn: Any = None
        if dsn:
            try:
                import psycopg

                self.conn = psycopg.connect(dsn, connect_timeout=3, autocommit=True)
                self.conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
                self.backend = "postgres"
            except Exception as e:  # unreachable DB must never break training
                self.error = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
                self.conn = None
        if self.conn is None:
            path = Path(sqlite_path or ARTIFACTS / "forgellm.db")
            path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(path, check_same_thread=False)
        self.schema = schema
        self.prefix = f"{schema}." if self.backend == "postgres" else ""
        self.ph = "%s" if self.backend == "postgres" else "?"
        for ddl in _SCHEMA:
            self._exec(ddl.format(s=self.prefix).replace("DOUBLE PRECISION", "REAL" if self.backend == "sqlite" else "DOUBLE PRECISION"))
        self._commit()

    def _exec(self, sql: str, params: tuple = ()) -> Any:
        return self.conn.execute(sql, params)

    def _commit(self) -> None:
        if self.backend == "sqlite":
            self.conn.commit()

    def _q(self, sql: str) -> str:
        return sql.replace("?", self.ph).replace("{t}", self.prefix)

    def describe(self) -> str:
        return f"{self.backend}" + (f" (postgres unavailable: {self.error})" if self.error else "")

    # -- runs ------------------------------------------------------------------------------------------
    def start_run(self, run_id: str, kind: str, name: str, config: dict[str, Any]) -> None:
        self._exec(self._q("DELETE FROM {t}runs WHERE id=?"), (run_id,))
        self._exec(self._q("INSERT INTO {t}runs VALUES (?,?,?,?,?,?,?,?)"),
                   (run_id, kind, name, json.dumps(config, default=str), "running", None, time.time(), None))
        self._commit()

    def log_metrics(self, run_id: str, step: int, payload: dict[str, Any]) -> None:
        self._exec(self._q("INSERT INTO {t}metrics VALUES (?,?,?)"), (run_id, step, json.dumps(payload, default=str)))
        self._commit()

    def finish_run(self, run_id: str, status: str, summary: dict[str, Any]) -> None:
        self._exec(self._q("UPDATE {t}runs SET status=?, summary=?, finished_at=? WHERE id=?"),
                   (status, json.dumps(summary, default=str), time.time(), run_id))
        self._commit()

    def runs(self, kind: str | None = None) -> list[dict[str, Any]]:
        cur = self._exec(self._q("SELECT id, kind, name, status, summary, started_at FROM {t}runs" + (" WHERE kind=?" if kind else "") + " ORDER BY started_at DESC"),
                         (kind,) if kind else ())
        return [{"id": r[0], "kind": r[1], "name": r[2], "status": r[3], "summary": json.loads(r[4]) if r[4] else None, "started_at": r[5]}
                for r in cur.fetchall()]

    def metrics(self, run_id: str) -> list[dict[str, Any]]:
        cur = self._exec(self._q("SELECT step, payload FROM {t}metrics WHERE run_id=? ORDER BY step"), (run_id,))
        return [{"step": s, **json.loads(p)} for s, p in cur.fetchall()]

    # -- registry --------------------------------------------------------------------------------------
    def register_adapter(self, name: str, base_model: str, method: str, task: str, path: str, num_parameters: int,
                         data_hash: str, config_hash: str, metrics: dict[str, Any]) -> int:
        cur = self._exec(self._q("SELECT COALESCE(MAX(version), 0) FROM {t}adapters WHERE name=?"), (name,))
        version = int(cur.fetchone()[0]) + 1
        self._exec(self._q("INSERT INTO {t}adapters VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"),
                   (name, version, base_model, method, task, path, "candidate", num_parameters, data_hash, config_hash,
                    json.dumps(metrics, default=str), None, time.time()))
        self._commit()
        return version

    def set_adapter_status(self, name: str, version: int, status: str, gate: dict[str, Any] | None = None) -> None:
        self._exec(self._q("UPDATE {t}adapters SET status=?, gate=? WHERE name=? AND version=?"),
                   (status, json.dumps(gate, default=str) if gate else None, name, version))
        self._commit()

    def adapters(self, name: str | None = None) -> list[dict[str, Any]]:
        cur = self._exec(self._q("SELECT name, version, base_model, method, task, path, status, num_parameters, data_hash, config_hash, metrics, gate, created_at FROM {t}adapters"
                                 + (" WHERE name=?" if name else "") + " ORDER BY name, version"), (name,) if name else ())
        cols = ["name", "version", "base_model", "method", "task", "path", "status", "num_parameters", "data_hash", "config_hash", "metrics", "gate", "created_at"]
        out = []
        for r in cur.fetchall():
            d = dict(zip(cols, r))
            d["metrics"] = json.loads(d["metrics"]) if d["metrics"] else {}
            d["gate"] = json.loads(d["gate"]) if d["gate"] else None
            out.append(d)
        return out

    def production_adapter(self, name: str) -> dict[str, Any] | None:
        rows = [a for a in self.adapters(name) if a["status"] == "production"]
        return rows[-1] if rows else None

    def save_eval(self, report_id: str, model_tag: str, suite: str, report: dict[str, Any]) -> None:
        self._exec(self._q("DELETE FROM {t}eval_reports WHERE id=?"), (report_id,))
        self._exec(self._q("INSERT INTO {t}eval_reports VALUES (?,?,?,?,?)"),
                   (report_id, model_tag, suite, json.dumps(report, default=str), time.time()))
        self._commit()

    def eval_reports(self, model_tag: str | None = None) -> list[dict[str, Any]]:
        cur = self._exec(self._q("SELECT id, model_tag, suite, report, created_at FROM {t}eval_reports" + (" WHERE model_tag=?" if model_tag else "") + " ORDER BY created_at"),
                         (model_tag,) if model_tag else ())
        return [{"id": r[0], "model_tag": r[1], "suite": r[2], "report": json.loads(r[3]), "created_at": r[4]} for r in cur.fetchall()]

    def import_sqlite(self, path: str | Path | None = None) -> dict[str, int]:
        """Copy rows from the local SQLite file into this store (used after a run that started before Postgres was up)."""
        src = Path(path or ARTIFACTS / "forgellm.db")
        if self.backend == "sqlite" or not src.exists():
            return {}
        lite = sqlite3.connect(src)
        counts: dict[str, int] = {}
        keys = {"runs": "id", "adapters": "name, version", "eval_reports": "id"}
        for table in ("runs", "metrics", "adapters", "eval_reports"):
            rows = lite.execute(f"SELECT * FROM {table}").fetchall()
            n = 0
            for r in rows:
                marks = ",".join([self.ph] * len(r))
                if table in keys:
                    cols = [c[1] for c in lite.execute(f"PRAGMA table_info({table})")]
                    where = " AND ".join(f"{c}={self.ph}" for c in keys[table].split(", "))
                    vals = [r[cols.index(c)] for c in keys[table].split(", ")]
                    if self._exec(f"SELECT 1 FROM {self.prefix}{table} WHERE {where}", tuple(vals)).fetchone():
                        continue
                self._exec(f"INSERT INTO {self.prefix}{table} VALUES ({marks})", tuple(r))
                n += 1
            counts[table] = n
        lite.close()
        self._commit()
        return counts

    def sync_adapters(self, root: str | Path | None = None) -> list[str]:
        """Register adapters that exist on disk but have no registry row (e.g. their rows lived in a store that is
        currently unreachable). Everything needed is in each adapter's manifest."""
        import json as _json

        root = Path(root or ARTIFACTS / "adapters")
        added = []
        for cfg in sorted(root.glob("*/adapter_config.json")):
            man = _json.loads(cfg.read_text())
            name = man.get("name", cfg.parent.name)
            if self.adapters(name):
                continue
            tasks = man.get("tasks") or []
            self.register_adapter(name, man.get("base_model", "?"), man.get("method", "lora"),
                                  tasks[0] if tasks else man.get("objective", "unknown"), str(cfg.parent),
                                  int(man.get("num_parameters", 0)), man.get("data_hash", ""), man.get("config_hash", ""),
                                  man.get("train_summary", {}))
            added.append(name)
        return added

    def drop_schema(self) -> None:
        """Postgres only: remove a scratch schema (used by tests)."""
        if self.backend == "postgres" and self.schema != "forgellm":
            self.conn.execute(f"DROP SCHEMA IF EXISTS {self.schema} CASCADE")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.conn.close()


_STORE: Store | None = None


def get_store() -> Store:
    global _STORE
    if _STORE is None:
        _STORE = Store()
    return _STORE
