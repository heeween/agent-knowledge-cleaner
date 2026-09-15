# Sanitized KB registry

`kb_revisions.jsonl` is generated from local state by `pipeline.py init`, review decisions, and publish. It contains only stable IDs, revision/status/supersedes and sanitized provenance. Formal question/answer text lives in immutable releases; this registry never contains raw paths, chats, credentials, or embeddings.

Do not edit generated records by hand. Existing `kb_id` values are permanent.
