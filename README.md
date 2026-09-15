# Agent Knowledge Cleaner

An offline, incremental knowledge-production pipeline for CRM chat exports. The legacy scripts and frozen outputs remain an immutable audit chain; new data is handled through `pipeline.py` without replaying that chain.

## Safety invariants

- `output/kb_entries_official_v3.jsonl` is the frozen 645-entry baseline. Its required SHA-256 is `f57b19feb7f6a6e118e7ab48737d54ce2f59d56a70c1f2fea9e274076561c51f`.
- Existing IDs `KB-0001..KB-0645` never change. New IDs start at `KB-0646`.
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

