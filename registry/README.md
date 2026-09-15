# Sanitized KB registry

`kb_revisions.jsonl` is generated from local state by `pipeline.py init`, review decisions, and publish. It is safe for Git only because it contains formal knowledge text and revision metadata—never raw paths, chats, credentials, or embeddings.

Do not edit generated records by hand. Existing `kb_id` values are permanent.
