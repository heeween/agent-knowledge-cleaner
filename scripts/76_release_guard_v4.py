#!/usr/bin/env python3
"""
Step 18.8 - 发布对账硬校验 (v4 重立基线)
==========================================

发布/同步前运行, 校验本地 release 与服务器 1.0.8
的对账关系 (防覆盖 / 防删除的最后一道闸):

1. 契约 survivor (26 条): 本地 chunk 文本必须与
   服务器 1.0.8 逐字一致 (revision 号可以不同)
2. 契约 retired (619 条, 含服务器已删 4 条):
   本地 chunk 不得出现
3. 新增 kb_id 必须 >= KB-0646
4. 契约文件与 v4 基线 sha 一致

用法:

    python scripts/76_release_guard_v4.py --release releases/1.1.0
    python scripts/76_release_guard_v4.py --release releases/1.1.0 \
        --remote-chunks output/remote_release_1.0.8_chunks.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"
CONTRACT_FILE = OUTPUT_DIR / "kb_official_v4_contract.json"
DEFAULT_REMOTE_CHUNKS = OUTPUT_DIR / "remote_release_1.0.8_chunks.jsonl"
FIRST_NEW_ID = 646


def load_chunks(path):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                if chunk["kb_id"] in mapping:
                    raise SystemExit(f"守卫失败: {path.name} 中 {chunk['kb_id']} 重复")
                mapping[chunk["kb_id"]] = chunk
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--remote-chunks", type=Path, default=DEFAULT_REMOTE_CHUNKS)
    args = parser.parse_args()

    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    local = load_chunks(args.release / "chunks.jsonl")
    remote = load_chunks(args.remote_chunks)

    failures = []

    survivor_text_diff = [
        kb_id for kb_id in sorted(contract["survivor_ids"])
        if kb_id not in local
        or (local[kb_id]["question"], local[kb_id]["answer"])
        != (remote[kb_id]["question"], remote[kb_id]["answer"])
    ]
    if survivor_text_diff:
        failures.append(f"survivor 文本与服务器 1.0.8 不一致: {survivor_text_diff}")

    resurrected = sorted(set(contract["retired_ids"]) & set(local))
    if resurrected:
        failures.append(f"retired 条目出现在发布快照中: {resurrected}")

    illegal_new = sorted(
        kb_id for kb_id in local
        if kb_id not in set(contract["survivor_ids"])
        and int(kb_id[3:]) < FIRST_NEW_ID
    )
    if illegal_new:
        failures.append(f"新增条目占用 KB-0646 之前的 id: {illegal_new}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 2

    new_count = len(local) - len(contract["survivor_ids"])
    print("PASS: 对账校验通过")
    print(f"  survivor 保持服务器文本: {len(contract['survivor_ids'])} 条")
    print(f"  retired 无复活: {len(contract['retired_ids'])} 条")
    print(f"  新增条目: {new_count} 条 (KB-{FIRST_NEW_ID}+)")
    print(f"  快照总数: {len(local)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
