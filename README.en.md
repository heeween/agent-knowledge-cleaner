# Agent Knowledge Cleaner

An offline, incremental knowledge-production pipeline for CRM chat exports. The legacy scripts and frozen outputs remain an immutable audit chain; new data is handled through `pipeline.py` without replaying that chain.

## Safety invariants

- `output/kb_entries_official_v3.jsonl` (645 entries) remains the immutable historical baseline for audit; it is no longer the publish baseline.
- Since 2026-09-20 the publish baseline is `output/kb_entries_official_v4.jsonl` (26 entries, required SHA-256 `a4080d3168903ad330a8156ee6cbba54648fc5a511a363fa7cb58e4873161029`): exactly the entries revised through the yj-kb kb-admin channel (texts frozen at server release 1.0.8). `output/kb_official_v4_contract.json` records survivor vs retired ids; `publish()` enforces it (survivors active, retired never active).
- The 619 retired v3 entries keep their revision/audit chains in the registry but never re-enter a snapshot. New knowledge only enters from chats after 2026-03-01 through the funnel v2 scripts (`scripts/69`–`76`), and new IDs start at `KB-0646`.
- Before any sync to yj-kb run `scripts/76_release_guard_v4.py --release releases/<version>`: it verifies survivor texts match server 1.0.8 verbatim and that no retired entry resurfaced.
- `.state/registry.sqlite3`, `data/`, `output/`, `.env`, and embeddings are not committed.
- Published releases contain sanitized QA data, not raw chat messages or timestamped Markdown.
- Text/status changes require review. A source change never silently rewrites a published KB revision.

## Commands

```bash
python pipeline.py init
python pipeline.py ingest /path/to/managed-chat-directory --dry-run
python pipeline.py ingest /path/to/managed-chat-directory
python pipeline.py review
python pipeline.py review --approve REV-... --note "verified against source"
python pipeline.py publish --version 1.0.1
python pipeline.py validate-release releases/1.0.1
python pipeline.py rollback
python scripts/build_release_embeddings.py releases/1.0.2
```

One-shot orchestration of the full release flow — publish → embeddings attach
→ remote preflight; `--apply` also stages remotely, verifies, switches
`current`, and hot-reloads. Completed steps are detected and skipped, so it is
safe to re-run:

```bash
python scripts/release.py --version 1.0.4             # 本地发布 + 向量挂载 + 远端只读预检
python scripts/release.py --version 1.0.4 --apply     # 全部步骤，含远端同步与热切换
python scripts/release.py --version 1.0.3 --sync-only # 本地已完成，只走远端
python scripts/release.py --version 1.0.5 --generate-missing --apply # 含修订/新增条目时补生成缺失向量
```

Directory ingest is authoritative for deletion detection inside that managed root. Single-file ingest never infers deletion of other files. The default overlap is 30 messages and can be changed with `--overlap-messages`.

Normal ingest uses the deterministic analyzer and never calls a network service. External analysis is opt-in:

```bash
OPENAI_API_KEY=... OPENAI_BASE_URL=https://provider.example/v1 OPENAI_MODEL=model-name \
  python pipeline.py ingest incoming/ --analyze --analyzer openai-compatible
```

Credentials are read only from environment variables. A dry-run will call an external analyzer only when both `--analyze` and `--analyzer openai-compatible` are supplied.

## State and recovery

The operational registry is local SQLite at `.state/registry.sqlite3`. Back it up using SQLite's online backup mechanism while the pipeline is stopped. Do not commit the backup: it contains paths and source messages. `registry/kb_revisions.jsonl` is the sanitized, Git-tracked identity/revision contract.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q pipeline.py incremental_kb tests
```

See [docs/YJ_KB_CONSUMER_CONTRACT.md](docs/YJ_KB_CONSUMER_CONTRACT.md) for the release format.

`build_release_embeddings.py` attaches an external `embeddings.jsonl` when every
published chunk has a vector. Default mode is fully offline: vectors come from the
frozen embedding cache under exact-text match, validated against the declared
model and dimension, bound to each KB revision and text SHA-256, and an existing
artifact is never overwritten. A new or revised chunk absent from the cache makes
the script refuse, unless `--generate-missing` is passed: such texts are embedded
via the declared model's API (`OPENAI_API_KEY`; optional `OPENAI_BASE_URL`) and
appended to a content-addressed supplement cache
(`output/rag_chunk_embeddings_supplement_v1.jsonl`), so subsequent runs reproduce
the artifact offline again. Vector files are excluded from Git and must accompany
the release when syncing to yj-kb.

## Syncing to yj-kb

The cleaner owns release synchronization. `scripts/sync_release.py` defaults to a read-only remote preflight. Only `--apply` stages an immutable version, verifies it on the host, switches `current`, and calls `/kb/reload`. If reload fails, it restores the previous pointer when available. It never deletes an old release.

```bash
python scripts/sync_release.py --version 1.0.2
python scripts/sync_release.py --version 1.0.2 --apply
python scripts/sync_release.py --rollback-to 1.0.0
python scripts/sync_release.py --rollback-to 1.0.0 --apply
```

Do not run `--apply` until the server has the new yj-kb consumer code and `/kb/status` and `/kb/reload` are available. The default remote root is `/root/yj-kb/cleaner_releases`, deliberately separate from the older yj-kb bootstrap `kb_releases`; the default host is `root@bk.rcar.vip`. Override with `--host` for another authorized host.
