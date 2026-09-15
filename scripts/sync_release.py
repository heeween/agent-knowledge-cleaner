#!/usr/bin/env python3
"""Safely stage a cleaner release on yj-kb host, verify it, and hot-switch.

Default is read-only preflight. --apply is required for any remote write.
Remote rollback is an explicit, separately validated operation.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from incremental_kb.core import validate_release  # noqa: E402
DEFAULT_HOST = "root@bk.rcar.vip"
DEFAULT_REMOTE_ROOT = "/root/yj-kb/cleaner_releases"
DEFAULT_PORT = 5006


def run_ssh(host: str, command: str, *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        input=input_bytes, capture_output=True, check=False,
    )


def require_remote_root(value: str) -> str:
    if not value.startswith("/root/yj-kb/") or ".." in Path(value).parts or value.endswith("/"):
        raise ValueError("remote root must be an explicit path below /root/yj-kb/")
    return value


def remote_preflight(host: str, remote_root: str, version: str) -> dict:
    quoted_root = shlex.quote(remote_root)
    quoted_version = shlex.quote(version)
    command = (
        f"if [ -d {quoted_root}/{quoted_version} ]; then "
        f"cd {quoted_root}/{quoted_version} && sha256sum -c SHA256SUMS; "
        f"else printf 'ABSENT\\n'; fi; "
        f"if [ -L {quoted_root}/current ]; then readlink {quoted_root}/current; "
        f"elif [ -f {quoted_root}/current ]; then cat {quoted_root}/current; fi"
    )
    result = run_ssh(host, command)
    if result.returncode:
        raise RuntimeError(f"read-only remote preflight failed: {result.stderr.decode(errors='replace')}")
    output = result.stdout.decode(errors="replace")
    return {"release_present": "ABSENT" not in output, "remote_output": output.strip()}


def require_remote_consumer_ready(host: str, port: int) -> None:
    result = run_ssh(host, f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/kb/status")
    if result.returncode or result.stdout.decode().strip() != "200":
        raise RuntimeError("remote /kb/status is not ready; deploy consumer code before any release write")


def stage_and_switch(host: str, remote_root: str, version: str, release: Path, port: int) -> dict:
    require_remote_consumer_ready(host, port)
    # Never overwrite an existing immutable version. The caller preflights first;
    # this guarded remote mkdir is the second race check.
    stage_name = f".{version}.stage-{uuid.uuid4().hex}"
    stage = f"{remote_root}/{stage_name}"
    qroot, qstage, qversion = map(shlex.quote, (remote_root, stage, version))
    qtarget = shlex.quote(f"{remote_root}/{version}")
    prepare = run_ssh(host, f"mkdir -p {qroot} && if [ -e {qtarget} ]; then exit 17; fi && mkdir {qstage}")
    if prepare.returncode:
        raise RuntimeError("remote release already exists or staging failed")
    tar = subprocess.run(["tar", "-czf", "-", "-C", str(release), "."], capture_output=True)
    if tar.returncode:
        raise RuntimeError("local tar archive failed")
    unpack = run_ssh(host, f"tar -xzf - -C {qstage}", input_bytes=tar.stdout)
    if unpack.returncode:
        raise RuntimeError("remote unpack failed; staged files retained for inspection")
    verify = run_ssh(host, f"cd {qstage} && sha256sum -c SHA256SUMS && test $(wc -l < chunks.jsonl) -eq $(python3 -c 'import json; print(json.load(open(\"manifest.json\"))[\"chunk_count\"])')")
    if verify.returncode:
        raise RuntimeError("remote SHA-256 or chunk count failed; current pointer unchanged")
    commit = run_ssh(host, f"if [ -e {qtarget} ]; then exit 17; fi && mv {qstage} {qtarget}")
    if commit.returncode:
        raise RuntimeError("remote immutable rename failed; current pointer unchanged")
    old = run_ssh(host, f"if [ -L {qroot}/current ]; then readlink {qroot}/current; fi")
    previous = old.stdout.decode().strip() if old.returncode == 0 else ""
    link = run_ssh(host, f"ln -s {qversion} {qroot}/.current.tmp && mv -Tf {qroot}/.current.tmp {qroot}/current")
    if link.returncode:
        raise RuntimeError("remote current switch failed; release staged but not activated")
    reload_result = run_ssh(host, f"curl -s -o /dev/null -w '%{{http_code}}' -X POST http://127.0.0.1:{port}/kb/reload")
    status = reload_result.stdout.decode().strip()
    if reload_result.returncode or not status.startswith("2"):
        if previous and re.fullmatch(r"\d+\.\d+\.\d+", previous):
            run_ssh(host, f"ln -s {shlex.quote(previous)} {qroot}/.current.tmp && mv -Tf {qroot}/.current.tmp {qroot}/current && curl -s -o /dev/null -X POST http://127.0.0.1:{port}/kb/reload")
        raise RuntimeError(f"hot reload failed (HTTP {status}); previous current pointer restored when available")
    return {"version": version, "previous": previous or None, "reload_http": status}


def remote_rollback(host: str, remote_root: str, version: str, port: int, *, apply: bool) -> dict:
    state = remote_preflight(host, remote_root, version)
    if not state["release_present"]:
        raise ValueError("requested rollback version is absent remotely")
    if not apply:
        return {"dry_run": True, "rollback_to": version, "preflight": state}
    require_remote_consumer_ready(host, port)
    qroot, qversion = map(shlex.quote, (remote_root, version))
    old = run_ssh(host, f"if [ -L {qroot}/current ]; then readlink {qroot}/current; fi")
    previous = old.stdout.decode().strip()
    switched = run_ssh(host, f"ln -s {qversion} {qroot}/.current.tmp && mv -Tf {qroot}/.current.tmp {qroot}/current && curl -s -o /dev/null -w '%{{http_code}}' -X POST http://127.0.0.1:{port}/kb/reload")
    status = switched.stdout.decode().strip()
    if switched.returncode or not status.startswith("2"):
        if previous and re.fullmatch(r"\d+\.\d+\.\d+", previous):
            run_ssh(host, f"ln -s {shlex.quote(previous)} {qroot}/.current.tmp && mv -Tf {qroot}/.current.tmp {qroot}/current")
        raise RuntimeError(f"rollback reload failed (HTTP {status})")
    return {"rollback_to": version, "previous": previous or None, "reload_http": status}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--version", help="release to sync; defaults to local current")
    parser.add_argument("--rollback-to", help="remote validated version to activate")
    parser.add_argument("--apply", action="store_true", help="permit remote writes and hot reload")
    args = parser.parse_args(argv)
    try:
        remote_root = require_remote_root(args.remote_root)
        version = args.rollback_to or args.version or (ROOT / "releases/current").resolve().name
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise ValueError("version must be SemVer")
        if args.rollback_to:
            result = remote_rollback(args.host, remote_root, version, args.port, apply=args.apply)
        else:
            release = ROOT / "releases" / version
            manifest = validate_release(release)
            state = remote_preflight(args.host, remote_root, version)
            result = {"dry_run": not args.apply, "version": version, "chunk_count": manifest["chunk_count"], "preflight": state}
            if args.apply:
                if state["release_present"]:
                    raise FileExistsError("remote immutable release already exists; use rollback-to or a new version")
                result = stage_and_switch(args.host, remote_root, version, release, args.port)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
