import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_release_embeddings", ROOT / "scripts/build_release_embeddings.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class BuildReleaseEmbeddingsTests(unittest.TestCase):
    def test_exact_text_identity_and_revision(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            release = root / "1.0.2"
            release.mkdir()
            (release / "manifest.json").write_text(json.dumps({
                "chunk_count": 1, "embedding": {"model": "embedding-3", "dimension": 2},
            }))
            (release / "chunks.jsonl").write_text(json.dumps({
                "chunk_id": "KB-0001@r2", "kb_id": "KB-0001", "revision": 2, "text": "问题：a\n答案：b",
            }) + "\n")
            source = root / "source.jsonl"
            source.write_text(json.dumps({"chunk_id": "KB-0001", "text": "问题：a\n答案：b"}) + "\n")
            vectors = root / "vectors.jsonl"
            vectors.write_text(json.dumps({"chunk_id": "KB-0001", "model": "embedding-3", "embedding": [0.1, 0.2]}) + "\n")

            records = builder.build_records(release, source, vectors)
            self.assertEqual(records[0]["chunk_id"], "KB-0001@r2")
            self.assertEqual(records[0]["content_sha256"], builder.content_hash("问题：a\n答案：b"))
            builder.write_artifact(release, records)
            with self.assertRaises(FileExistsError):
                builder.write_artifact(release, records)
            source.write_text(json.dumps({"chunk_id": "KB-0001", "text": "stale"}) + "\n")
            with self.assertRaisesRegex(ValueError, "exact-text"):
                builder.build_records(release, source, vectors)


if __name__ == "__main__":
    unittest.main()
