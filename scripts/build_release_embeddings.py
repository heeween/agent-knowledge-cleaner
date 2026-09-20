#!/usr/bin/env python3
"""Attach a complete, content-bound external embedding artifact to a cleaner release.

Default mode stays fully offline: vectors come from the frozen RAG embedding cache
only when model, dimension and exact text match the release. It never calls an API
or changes an existing release artifact.

With --generate-missing, chunks absent from the frozen cache (e.g. revised
KB-0001..KB-0645 text or new entries) are embedded through the declared model's
API (OPENAI_API_KEY required; OPENAI_BASE_URL optional, defaults to bigmodel).
Generated vectors are appended to a content-addressed supplement cache so later
runs reproduce the artifact offline again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checked_vector(vector: object, model: str, dimension: int, chunk_id: str) -> list:
    if not isinstance(vector, list) or len(vector) != dimension:
        raise ValueError(f"{chunk_id} dimension mismatch")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
        raise ValueError(f"{chunk_id} has invalid vector values")
    return vector


def api_generator(model: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("--generate-missing requires the OPENAI_API_KEY environment variable")
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=0)

    def generate(text: str) -> list:
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.embeddings.create(model=model, input=text)
                return response.data[0].embedding
            except Exception as error:
                last_error = error
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"embedding request failed after {MAX_RETRIES} attempts: {last_error}")

    return generate


def load_supplement(path: Path, model: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    supplement = {}
    for item in read_jsonl(path):
        if item.get("model") == model:
            supplement[item["content_sha256"]] = item
    return supplement


def build_records(release: Path, source_chunks: Path, source_vectors: Path, *,
                  supplement: dict[str, dict] | None = None,
                  generate=None, generated_out: list[str] | None = None) -> list[dict]:
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
        chunk_id = chunk["chunk_id"]
        source = source_by_id.get(kb_id)
        cached = vectors_by_id.get(kb_id)
        vector = None
        if source and source.get("text") == chunk["text"] and cached:
            if cached.get("model") != model:
                raise ValueError(f"{chunk_id} model mismatch")
            vector = checked_vector(cached.get("embedding"), model, dimension, chunk_id)
        else:
            item = (supplement or {}).get(content_hash(chunk["text"]))
            if item is not None:
                if item.get("model") != model:
                    raise ValueError(f"{chunk_id} supplement model mismatch")
                vector = checked_vector(item.get("embedding"), model, dimension, chunk_id)
            elif generate is not None:
                vector = checked_vector(generate(chunk["text"]), model, dimension, chunk_id)
                if generated_out is not None:
                    generated_out.append(content_hash(chunk["text"]))
            else:
                raise ValueError(f"{chunk_id} has no exact-text cached vector")
        records.append({
            "chunk_id": chunk_id, "kb_id": kb_id,
            "revision": chunk["revision"], "model": model,
            "content_sha256": content_hash(chunk["text"]), "embedding": vector,
        })
    if len(records) != len(chunks):
        raise ValueError("incomplete vector coverage")
    return records


def append_supplement(path: Path, records: list[dict], generated: list[str]) -> None:
    generated_set = set(generated)
    pending = [
        {"content_sha256": record["content_sha256"], "model": record["model"], "embedding": record["embedding"]}
        for record in records if record["content_sha256"] in generated_set
    ]
    if not pending:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        for item in pending:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"supplement_appended": len(pending), "path": str(path)}, ensure_ascii=False))


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
    parser.add_argument("--supplement", type=Path, default=ROOT / "output/rag_chunk_embeddings_supplement_v1.jsonl",
                        help="content-addressed cache for vectors generated outside the frozen cache")
    parser.add_argument("--generate-missing", action="store_true",
                        help="call the embedding API for chunks absent from cache/supplement (default: refuse)")
    args = parser.parse_args()
    release = args.release.resolve()

    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    model = manifest["embedding"]["model"]
    supplement = load_supplement(args.supplement, model)
    generate = api_generator(model) if args.generate_missing else None
    generated: list[str] = []
    records = build_records(release, args.source_chunks, args.source_vectors,
                            supplement=supplement, generate=generate, generated_out=generated)
    if generated:
        append_supplement(args.supplement, records, generated)
    target = write_artifact(release, records)
    from_supplement = len({record["content_sha256"] for record in records} & set(supplement) - set(generated))
    print(json.dumps({"artifact": str(target), "vectors": len(records), "model": model,
                      "from_frozen_cache": len(records) - len(generated) - from_supplement,
                      "from_supplement": from_supplement, "generated": len(generated)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
