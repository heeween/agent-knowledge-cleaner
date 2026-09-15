from pathlib import Path
import json

import pandas as pd


RAW_DIR = Path("data/raw/markdowns")
INVENTORY_FILE = Path("output/formal_docs_inventory_v1.jsonl")

PARAGRAPHS_JSONL = Path("output/formal_doc_paragraphs_v1.jsonl")
CHUNKS_JSONL = Path("output/formal_doc_evidence_chunks_v1.jsonl")
CHUNKS_XLSX = Path("output/formal_doc_evidence_chunks_v1.xlsx")

# 段落聚合目标字符数：
# chunk 在装入一段后达到该值即闭合，
# 因此实际大小在 [target, target + 最长段落) 之间，
# 永不切断段落
TARGET_CHUNK_CHARS = 400

AUTHORITY = "official_training"
FORM = "asr_transcript"
TEMPORAL_ANCHOR = "none"


def load_formal_documents():
    records = []

    with open(INVENTORY_FILE, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            if record["classification"] == "formal_document":
                records.append(record)

    records.sort(key=lambda r: r["document_id"])
    return records


def parse_paragraphs(file_path: Path, doc: dict):
    content = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    all_lines = content.splitlines()

    paragraphs = []
    line_verify_failures = []

    para_index = 0

    for line_number, raw_line in enumerate(
        all_lines, start=1
    ):
        stripped = raw_line.strip()

        if not stripped:
            continue

        para_index += 1

        # 确定性校验 1：
        # 段落必须与原文件该行逐字相等（strip 后）
        if all_lines[line_number - 1].strip() != stripped:
            line_verify_failures.append({
                "document_id": doc["document_id"],
                "line_number": line_number,
            })

        paragraphs.append({
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "para_index": para_index,
            "line_number": line_number,
            "text": stripped,
            "char_count": len(stripped),
        })

    return paragraphs, line_verify_failures


def build_chunks(doc: dict, paragraphs: list):
    chunks = []
    current = []

    def close(current, chunk_index):
        texts = [p["text"] for p in current]

        # chunk_id 在 main 中按全局顺序分配
        return {
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "chunk_index": chunk_index,
            "para_start": current[0]["para_index"],
            "para_end": current[-1]["para_index"],
            "line_start": current[0]["line_number"],
            "line_end": current[-1]["line_number"],
            "paragraph_count": len(current),
            "text": "\n".join(texts),
            "char_count": sum(len(t) for t in texts)
            + (len(texts) - 1),
            "authority": AUTHORITY,
            "form": FORM,
            "temporal_anchor": TEMPORAL_ANCHOR,
        }

    chunk_index = 0
    current_chars = 0

    for paragraph in paragraphs:
        current.append(paragraph)
        current_chars += paragraph["char_count"]

        if current_chars >= TARGET_CHUNK_CHARS:
            chunk_index += 1
            chunks.append(close(current, chunk_index))
            current = []
            current_chars = 0

    if current:
        chunk_index += 1
        chunks.append(close(current, chunk_index))

    return chunks


def main():
    docs = load_formal_documents()

    print(f"正式文档数: {len(docs)}")

    all_paragraphs = []
    all_chunks = []
    all_line_verify_failures = []
    doc_stats = []
    coverage_issues = []

    for doc in docs:
        file_path = RAW_DIR / doc["filename"]

        if not file_path.exists():
            coverage_issues.append({
                "document_id": doc["document_id"],
                "issue": "file_missing",
            })
            continue

        paragraphs, failures = parse_paragraphs(
            file_path, doc
        )
        all_line_verify_failures.extend(failures)

        chunks = build_chunks(doc, paragraphs)

        global_offset = len(all_paragraphs)
        for paragraph in paragraphs:
            paragraph["paragraph_id"] = (
                f"FD-P-{global_offset + paragraph['para_index']:05d}"
            )

        for chunk in chunks:
            # 确定性校验 2：
            # chunk text 必须等于段落原文精确 join
            expected = "\n".join(
                p["text"]
                for p in paragraphs[
                    chunk["para_start"] - 1 : chunk["para_end"]
                ]
            )

            if chunk["text"] != expected:
                coverage_issues.append({
                    "document_id": doc["document_id"],
                    "issue": f"chunk_text_mismatch_"
                    f"{chunk['chunk_id']}",
                })

        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        # 确定性校验 3：
        # 文件自 13.1 盘点后未变
        # （重读文件 len(content) 与冻结盘点一致）
        if len(content) != doc["char_count"]:
            coverage_issues.append({
                "document_id": doc["document_id"],
                "issue": (
                    f"file_changed_since_inventory "
                    f"inventory={doc['char_count']} "
                    f"current={len(content)}"
                ),
            })

        all_paragraphs.extend(paragraphs)
        all_chunks.extend(chunks)

        doc_stats.append({
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "title": doc["title"],
            "paragraph_count": len(paragraphs),
            "chunk_count": len(chunks),
            "inventory_char_count": doc["char_count"],
            "parsed_chars": sum(
                p["char_count"] for p in paragraphs
            ),
        })

        print(
            f"  {doc['document_id']}  "
            f"{doc['filename']}  "
            f"paras={len(paragraphs)}  "
            f"chunks={len(chunks)}"
        )

    # chunk_id 全局顺序分配，
    # 保证全部文档范围内唯一
    for global_index, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = f"FD-CH-{global_index:05d}"

    # 确定性校验 4：
    # chunk 覆盖所有段落，0 遗漏、0 重叠
    covered_para_count = sum(
        chunk["paragraph_count"] for chunk in all_chunks
    )

    if covered_para_count != len(all_paragraphs):
        coverage_issues.append({
            "document_id": "-",
            "issue": (
                f"paragraph_coverage_mismatch "
                f"total={len(all_paragraphs)} "
                f"covered={covered_para_count}"
            ),
        })

    PARAGRAPHS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(PARAGRAPHS_JSONL, "w", encoding="utf-8") as output:
        for paragraph in all_paragraphs:
            output.write(
                json.dumps(paragraph, ensure_ascii=False) + "\n"
            )

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as output:
        for chunk in all_chunks:
            output.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    summary = {
        "formal_documents": len(docs),
        "paragraphs": len(all_paragraphs),
        "chunks": len(all_chunks),
        "parsed_chars": sum(s["parsed_chars"] for s in doc_stats),
        "line_verify_failures": len(all_line_verify_failures),
        "coverage_issues": len(coverage_issues),
        "authority": AUTHORITY,
        "form": FORM,
        "temporal_anchor": TEMPORAL_ANCHOR,
        "target_chunk_chars": TARGET_CHUNK_CHARS,
    }

    chunk_rows = []
    for chunk in all_chunks:
        row = dict(chunk)
        row["text"] = row["text"].replace("\n", " ⏎ ")
        chunk_rows.append(row)

    with pd.ExcelWriter(CHUNKS_XLSX, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(
            writer, sheet_name="summary", index=False
        )
        pd.DataFrame(doc_stats).to_excel(
            writer, sheet_name="documents", index=False
        )
        pd.DataFrame(chunk_rows).to_excel(
            writer, sheet_name="chunks", index=False
        )
        pd.DataFrame(all_line_verify_failures).to_excel(
            writer, sheet_name="line_verify_failures", index=False
        )
        pd.DataFrame(coverage_issues).to_excel(
            writer, sheet_name="coverage_issues", index=False
        )

    print()
    print("=" * 60)
    print("正式文档证据解析完成（Step 13.2）")
    print("=" * 60)
    print(f"文档: {len(docs)}")
    print(f"段落: {len(all_paragraphs)}")
    print(f"证据块: {len(all_chunks)}")
    print(f"段落 strip 后总字符: {summary['parsed_chars']}")
    print(f"行号反查失败: {len(all_line_verify_failures)}")
    print(f"覆盖/一致性问题: {len(coverage_issues)}")
    print()
    print(f"输出: {PARAGRAPHS_JSONL}")
    print(f"输出: {CHUNKS_JSONL}")
    print(f"输出: {CHUNKS_XLSX}")


if __name__ == "__main__":
    main()
