#!/usr/bin/env python3
"""Attach a complete, content-bound external embedding artifact to a cleaner release.

Reuses the frozen RAG embedding cache only when model, dimension and exact text
match the release. It never calls an API or changes an existing release artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_records(release: Path, source_chunks: Path, source_vectors: Path) -> list[dict]:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    model = manifest["embedding"]["model"]
    dimension = manifest["embedding"]["dimension"]
    chunks = read_jsonl(release / "chunks.jsonl")
    if len(chunks) != manifest["chunk_count"]:
        raise ValueError("release chunk_count mismatch")

    source_by_id = {}
    for item in read_jsonl(source_chunks):
        old_id = item["chunk_id"]
        if old_id in source_by_id:
            raise ValueError(f"duplicate source chunk: {old_id}")
        source_by_id[old_id] = item
    vectors_by_id = {}
    for item in read_jsonl(source_vectors):
        old_id = item["chunk_id"]
        if old_id in vectors_by_id:
            raise ValueError(f"duplicate source embedding: {old_id}")
        vectors_by_id[old_id] = item

    records = []
    for chunk in chunks:
        kb_id = chunk["kb_id"]
        source = source_by_id.get(kb_id)
        cached = vectors_by_id.get(kb_id)
        if not source or source.get("text") != chunk["text"] or not cached:
            raise ValueError(f"{chunk['chunk_id']} has no exact-text cached vector")
        vector = cached.get("embedding")
        if cached.get("model") != model or not isinstance(vector, list) or len(vector) != dimension:
            raise ValueError(f"{chunk['chunk_id']} model/dimension mismatch")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
            raise ValueError(f"{chunk['chunk_id']} has invalid vector values")
        records.append({
            "chunk_id": chunk["chunk_id"], "kb_id": kb_id,
            "revision": chunk["revision"], "model": model,
            "content_sha256": content_hash(chunk["text"]), "embedding": vector,
        })
    if len(records) != len(chunks):
        raise ValueError("incomplete vector coverage")
    return records


def write_artifact(release: Path, records: list[dict]) -> Path:
    target = release / "embeddings.jsonl"
    if target.exists():
        raise FileExistsError(f"external vector artifact already exists: {target}")
    fd, temporary = tempfile.mkstemp(prefix=".embeddings-", suffix=".jsonl", dir=release)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, target)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--source-chunks", type=Path, default=ROOT / "output/rag_chunks_v1.jsonl")
    parser.add_argument("--source-vectors", type=Path, default=ROOT / "output/rag_chunk_embeddings_v1.jsonl")
    args = parser.parse_args()
    release = args.release.resolve()
    records = build_records(release, args.source_chunks, args.source_vectors)
    target = write_artifact(release, records)
    print(json.dumps({"artifact": str(target), "vectors": len(records), "model": records[0]["model"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
