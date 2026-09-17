#!/usr/bin/env python3
"""一键发布编排：publish → 挂载向量 → 远端同步。

按顺序调用三个既有守卫脚本，不替代它们、不绕过任何守卫：

  1. pipeline.py publish --version X       本地不可变快照
                                           （干净树 / 冻结基线 / 原子落盘）
  2. scripts/build_release_embeddings.py   从冻结缓存挂载 embeddings.jsonl
                                           （无 API；逐条文本/模型/维度校验）
  3. scripts/sync_release.py --version X   远端只读预检；--apply 才
                                           暂存 + SHA 校验 + 原子切换 + 热加载

已完成步骤自动检测并跳过：本地 release 存在且通过 validate_release、
embeddings 已挂载且逐条内容哈希相符、远端版本已存在。可安全重跑。

用法（建议用 .venv/bin/python 运行）：

  python scripts/release.py --version 1.0.4             # 本地两步 + 远端只读预检
  python scripts/release.py --version 1.0.4 --apply     # 全部步骤含远端同步热切换
  python scripts/release.py --version 1.0.3 --sync-only # 本地已完成，只走远端
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from incremental_kb.core import validate_release  # noqa: E402


def run(title: str, cmd: list[str]) -> None:
    print(f"\n=== {title} ===")
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode:
        print(f"!! {title} 失败（exit {result.returncode}）", file=sys.stderr)
        sys.exit(result.returncode)


def check_local_release(release: Path) -> bool:
    """release 已存在时校验其完整性；不存在返回 False。"""
    if not (release / "manifest.json").is_file():
        return False
    validate_release(release)
    print(f"[skip] {release.name} 已存在且 validate_release 通过")
    return True


def check_embeddings(release: Path) -> bool:
    """embeddings.jsonl 已挂载且与 chunks 逐条相符时返回 True。"""
    artifact = release / "embeddings.jsonl"
    if not artifact.is_file():
        return False
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    model = manifest["embedding"]["model"]
    dimension = manifest["embedding"]["dimension"]
    chunks = {
        json.loads(line)["chunk_id"]: json.loads(line)
        for line in (release / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    seen = 0
    for line in artifact.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        chunk = chunks.get(record["chunk_id"])
        if (
            chunk is None
            or record["kb_id"] != chunk["kb_id"]
            or record["revision"] != chunk["revision"]
            or record["model"] != model
            or len(record["embedding"]) != dimension
            or record["content_sha256"]
            != hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()
        ):
            raise ValueError(f"embeddings.jsonl 与 chunks 不符: {record.get('chunk_id')}")
        seen += 1
    if seen != len(chunks):
        raise ValueError(f"向量覆盖不全: {seen}/{len(chunks)}")
    print(f"[skip] embeddings.jsonl 已挂载且 {seen} 条逐项相符")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="SemVer 版本号，如 1.0.4")
    parser.add_argument("--apply", action="store_true", help="允许远端写入与热切换")
    parser.add_argument("--sync-only", action="store_true", help="跳过本地两步，只做远端同步")
    args = parser.parse_args(argv)

    release = ROOT / "releases" / args.version
    python = sys.executable

    if not args.sync_only:
        if not check_local_release(release):
            run("步骤 1/3 本地发布", [python, str(ROOT / "pipeline.py"), "publish", "--version", args.version])
        if not check_embeddings(release):
            run(
                "步骤 2/3 挂载向量制品",
                [python, str(ROOT / "scripts" / "build_release_embeddings.py"), str(release)],
            )

    run("步骤 3/3 远端预检（只读）", [python, str(ROOT / "scripts" / "sync_release.py"), "--version", args.version])
    if args.apply:
        run("步骤 3/3 远端同步与热切换", [python, str(ROOT / "scripts" / "sync_release.py"), "--version", args.version, "--apply"])

    print(f"\n完成：releases/{args.version}"
          + ("（已同步远端并热切换）" if args.apply else "（本地就绪；远端写入需 --apply）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
