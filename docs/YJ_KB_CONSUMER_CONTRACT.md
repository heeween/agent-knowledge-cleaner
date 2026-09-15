# yj-kb consumer contract

This document defines the handoff only. It does not authorize changes or deployment to `yj-kb`.

## Release discovery and switching

`releases/current` is a relative symlink to an immutable SemVer directory such as `releases/1.0.0`. A consumer should:

1. Resolve `current` once and open that version directory.
2. Read `manifest.json` and reject unsupported `schema_version` values.
3. Verify every digest in `manifest.files` and verify `chunk_count` against `chunks.jsonl`.
4. Load all chunks into a new in-memory/index generation.
5. Atomically swap the active generation only after the complete load succeeds.
6. Keep the old generation available until the swap is confirmed.

Never combine files from different release directories. `changelog.json` is for audit and display; it is not required to reconstruct the active snapshot.

## Manifest schema 1.0.0

Required fields:

- `schema_version`: release contract version, independent from content version.
- `release_version`: immutable SemVer release directory name.
- `generated_at`: UTC RFC 3339 timestamp.
- `chunk_count`: number of JSONL records.
- `embedding.model` and `embedding.dimension`: expected regenerable embedding configuration.
- `embedding.artifact`: `external_regenerable`; vectors are not in the Git release.
- `source_commit`: Git commit used to generate the release.
- `files`: SHA-256 descriptor for `chunks.jsonl` and `changelog.json`.

## Chunk schema 1.0.0

Each JSONL record contains:

- `chunk_id`: revision-specific identity, `{kb_id}@r{revision}`.
- `kb_id`: stable logical knowledge identity.
- `revision`: monotonically increasing integer within a `kb_id`.
- `status`: `active` in the complete published snapshot.
- `question`, `answer`, and retrieval `text` (`问题：...\n答案：...`).
- `supersedes`: prior revision identity or null.
- `provenance`: sanitized audit reference; never a raw path or chat body.

This is structured QA. It must not be rewritten as legacy timestamped chat Markdown. Consumers should store `kb_id`, `revision`, and `chunk_id` with vectors so answers can cite the exact published revision.

## Rollback

Rollback changes only `releases/current` to an already validated immutable directory. Consumers should treat the pointer change exactly like a new release and atomically reload it. No release directory is deleted by rollback.

