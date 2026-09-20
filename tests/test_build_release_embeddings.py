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

    def test_generate_missing_persists_supplement_for_offline_rerun(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            release = root / "1.0.5"
            release.mkdir()
            (release / "manifest.json").write_text(json.dumps({
                "chunk_count": 1, "embedding": {"model": "embedding-3", "dimension": 2},
            }))
            (release / "chunks.jsonl").write_text(json.dumps({
                "chunk_id": "KB-0004@r2", "kb_id": "KB-0004", "revision": 2, "text": "问题：a\n答案：修正后的两步走答案",
            }) + "\n")
            source = root / "source.jsonl"
            source.write_text(json.dumps({"chunk_id": "KB-0004", "text": "问题：a\n答案：旧答案"}) + "\n")
            vectors = root / "vectors.jsonl"
            vectors.write_text(json.dumps({"chunk_id": "KB-0004", "model": "embedding-3", "embedding": [0.1, 0.2]}) + "\n")

            def fake_generate(text: str) -> list:
                self.assertIn("修正后", text)
                return [0.3, 0.4]

            generated = []
            records = builder.build_records(release, source, vectors, supplement={}, generate=fake_generate, generated_out=generated)
            self.assertEqual(records[0]["chunk_id"], "KB-0004@r2")
            self.assertEqual(records[0]["embedding"], [0.3, 0.4])
            self.assertEqual(generated, [records[0]["content_sha256"]])

            supplement_path = root / "supplement.jsonl"
            builder.append_supplement(supplement_path, records, generated)
            reloaded = builder.load_supplement(supplement_path, "embedding-3")
            self.assertEqual(reloaded[records[0]["content_sha256"]]["embedding"], [0.3, 0.4])

            offline = builder.build_records(release, source, vectors, supplement=reloaded)
            self.assertEqual(offline[0]["embedding"], [0.3, 0.4])

            def bad_generate(text: str) -> list:
                return [0.5]

            with self.assertRaisesRegex(ValueError, "dimension mismatch"):
                builder.build_records(release, source, vectors, supplement={}, generate=bad_generate)


if __name__ == "__main__":
    unittest.main()
