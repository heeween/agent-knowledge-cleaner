from pathlib import Path
import hashlib
import json
import re

import pandas as pd


RAW_DIR = Path("data/raw/markdowns")
MESSAGES_FILE = Path("output/messages.jsonl")
FROZEN_INVENTORY = Path("output/document_inventory.xlsx")

OUTPUT_JSONL = Path("output/formal_docs_inventory_v1.jsonl")
OUTPUT_XLSX = Path("output/formal_docs_inventory_v1.xlsx")

# 与 03_parse_chat.py 完全相同的聊天行模式，
# 用于二次确认这些文档没有被聊天解析覆盖
MESSAGE_PATTERN = re.compile(
    r"^【(?P<timestamp>[^】]+)】(?P<speaker>[^:：]+)[:：]\s*(?P<content>.*)$"
)

DATE_PATTERN = re.compile(
    r"(?:20\d{2}[-/年.]\s*\d{1,2}[-/月.]\s*\d{1,2}日?|20\d{2}[-/年])"
)
VERSION_PATTERN = re.compile(
    r"[Vv](?:\d+\.)+\d+|\d+\.\d+\.\d+"
)


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def extract_title(content: str, filename: str) -> str:
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)

    if match:
        return match.group(1).strip()

    return filename


def classify_document(non_empty_lines: list, chat_line_count: int):
    # 非聊天解析覆盖的文件中，
    # 只有标题一行的空壳（聊天导出但零消息）
    # 不能当作正式文档使用
    if len(non_empty_lines) <= 1:
        return "empty_stub"

    if chat_line_count > 0:
        return "chat_content_detected"

    return "formal_document"


def structure_stats(content: str, lines: list):
    headings = []
    table_lines = 0
    image_count = 0
    link_count = 0
    code_fence_count = 0
    list_line_count = 0
    paragraph_count = 0
    date_mentions = []
    version_mentions = []

    in_code_block = False

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("```"):
            in_code_block = not in_code_block
            code_fence_count += 1
            continue

        if in_code_block:
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)

        if heading_match:
            headings.append({
                "line_number": line_number,
                "level": len(heading_match.group(1)),
                "text": heading_match.group(2).strip(),
            })
            continue

        if line.startswith("|"):
            table_lines += 1
            continue

        if re.match(r"^[-*+]\s+", line) or re.match(r"^\d+[.、)]\s+", line):
            list_line_count += 1
            continue

        paragraph_count += 1

        image_count += len(re.findall(r"!\[[^\]]*\]\([^)]+\)", line))
        link_count += len(
            re.findall(r"(?<!\!)\[[^\]]+\]\([^)]+\)", line)
        )

        for match in DATE_PATTERN.finditer(line):
            date_mentions.append({
                "line_number": line_number,
                "text": match.group(0),
                "context": line[:120],
            })

        for match in VERSION_PATTERN.finditer(line):
            version_mentions.append({
                "line_number": line_number,
                "text": match.group(0),
                "context": line[:120],
            })

    return {
        "headings": headings,
        "table_lines": table_lines,
        "image_count": image_count,
        "link_count": link_count,
        "code_fence_count": code_fence_count,
        "list_line_count": list_line_count,
        "paragraph_count": paragraph_count,
        "date_mentions": date_mentions,
        "version_mentions": version_mentions,
    }


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"目录不存在: {RAW_DIR}")

    files = sorted(RAW_DIR.glob("*.md"))

    chat_doc_ids = set()

    with open(MESSAGES_FILE, encoding="utf-8") as f:
        for line in f:
            chat_doc_ids.add(json.loads(line)["document_id"])

    frozen = pd.read_excel(FROZEN_INVENTORY, dtype=str)
    frozen_index = {
        row["document_id"]: row
        for _, row in frozen.iterrows()
    }

    records = []
    crosscheck_mismatches = []

    for index, file_path in enumerate(files, start=1):
        document_id = f"D{index:04d}"

        if document_id in chat_doc_ids:
            continue

        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        all_lines = content.splitlines()
        non_empty_lines = [
            line for line in all_lines if line.strip()
        ]

        chat_line_count = sum(
            1
            for line in non_empty_lines
            if MESSAGE_PATTERN.match(line.strip())
        )

        sha256 = calculate_sha256(file_path)

        classification = classify_document(
            non_empty_lines,
            chat_line_count
        )

        stats = structure_stats(content, all_lines)

        record = {
            "document_id": document_id,
            "filename": file_path.name,
            "relative_path": str(
                file_path.relative_to(RAW_DIR)
            ),
            "file_size": file_path.stat().st_size,
            "sha256": sha256,
            "title": extract_title(content, file_path.name),
            "classification": classification,
            "non_empty_line_count": len(non_empty_lines),
            "chat_pattern_line_count": chat_line_count,
            "char_count": len(content),
            "heading_count": len(stats["headings"]),
            "table_lines": stats["table_lines"],
            "image_count": stats["image_count"],
            "link_count": stats["link_count"],
            "code_fence_count": stats["code_fence_count"],
            "list_line_count": stats["list_line_count"],
            "paragraph_count": stats["paragraph_count"],
            "date_mention_count": len(stats["date_mentions"]),
            "version_mention_count": len(
                stats["version_mentions"]
            ),
            "headings": stats["headings"],
            "date_mentions": stats["date_mentions"],
            "version_mentions": stats["version_mentions"],
        }

        records.append(record)

        frozen_row = frozen_index.get(document_id)

        if frozen_row is None:
            crosscheck_mismatches.append({
                "document_id": document_id,
                "field": "presence",
                "frozen": "MISSING",
                "recomputed": "present",
            })
        else:
            if str(frozen_row["filename"]) != file_path.name:
                crosscheck_mismatches.append({
                    "document_id": document_id,
                    "field": "filename",
                    "frozen": str(frozen_row["filename"]),
                    "recomputed": file_path.name,
                })

            if str(frozen_row["sha256"]) != sha256:
                crosscheck_mismatches.append({
                    "document_id": document_id,
                    "field": "sha256",
                    "frozen": str(frozen_row["sha256"]),
                    "recomputed": sha256,
                })

    # 冻结盘点表中不存在于当前非聊天集合的 document_id
    local_ids = {record["document_id"] for record in records}

    for document_id in frozen_index:
        if document_id in local_ids:
            continue

        if document_id in chat_doc_ids:
            continue

        crosscheck_mismatches.append({
            "document_id": document_id,
            "field": "presence",
            "frozen": str(frozen_index[document_id]["filename"]),
            "recomputed": "MISSING in non-chat set",
        })

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    formal = [
        record for record in records
        if record["classification"] == "formal_document"
    ]
    stubs = [
        record for record in records
        if record["classification"] == "empty_stub"
    ]
    chat_contaminated = [
        record for record in records
        if record["classification"] == "chat_content_detected"
    ]

    flat_rows = []
    heading_rows = []
    date_rows = []
    version_rows = []

    for record in records:
        flat_rows.append({
            key: value
            for key, value in record.items()
            if key not in (
                "headings", "date_mentions", "version_mentions"
            )
        })

        for heading in record["headings"]:
            heading_rows.append({
                "document_id": record["document_id"],
                "filename": record["filename"],
                **heading,
            })

        for mention in record["date_mentions"]:
            date_rows.append({
                "document_id": record["document_id"],
                "filename": record["filename"],
                **mention,
            })

        for mention in record["version_mentions"]:
            version_rows.append({
                "document_id": record["document_id"],
                "filename": record["filename"],
                **mention,
            })

    flat_df = pd.DataFrame(flat_rows)
    formal_df = flat_df[
        flat_df["classification"] == "formal_document"
    ].copy()
    stub_df = flat_df[
        flat_df["classification"] == "empty_stub"
    ].copy()

    summary = {
        "total_markdown_files": len(files),
        "chat_documents": len(chat_doc_ids),
        "non_chat_documents": len(records),
        "formal_documents": len(formal),
        "empty_stubs": len(stubs),
        "chat_content_detected": len(chat_contaminated),
        "frozen_inventory_mismatches": len(
            crosscheck_mismatches
        ),
        "formal_doc_char_total": int(
            formal_df["char_count"].sum()
        ) if len(formal_df) else 0,
        "formal_doc_date_mention_total": int(
            formal_df["date_mention_count"].sum()
        ) if len(formal_df) else 0,
        "formal_doc_version_mention_total": int(
            formal_df["version_mention_count"].sum()
        ) if len(formal_df) else 0,
    }

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(
            writer, sheet_name="summary", index=False
        )
        flat_df.to_excel(
            writer, sheet_name="all_non_chat", index=False
        )
        formal_df.to_excel(
            writer, sheet_name="formal_documents", index=False
        )
        stub_df.to_excel(
            writer, sheet_name="empty_stubs", index=False
        )
        pd.DataFrame(heading_rows).to_excel(
            writer, sheet_name="heading_outline", index=False
        )
        pd.DataFrame(date_rows).to_excel(
            writer, sheet_name="date_mentions", index=False
        )
        pd.DataFrame(version_rows).to_excel(
            writer, sheet_name="version_mentions", index=False
        )
        pd.DataFrame(crosscheck_mismatches).to_excel(
            writer, sheet_name="frozen_crosscheck", index=False
        )

    print("=" * 60)
    print("正式文档盘点完成（Step 13.1）")
    print("=" * 60)
    print(f"Markdown 总数: {len(files)}")
    print(f"聊天文档: {len(chat_doc_ids)}")
    print(f"非聊天文档: {len(records)}")
    print(f"  formal_document: {len(formal)}")
    print(f"  empty_stub: {len(stubs)}")
    print(f"  chat_content_detected: {len(chat_contaminated)}")
    print(
        "冻结 document_inventory 交叉校验不一致: "
        f"{len(crosscheck_mismatches)}"
    )
    print()
    print("正式文档清单:")

    for record in formal:
        print(
            f"  {record['document_id']}  "
            f"{record['filename']}  "
            f"chars={record['char_count']}  "
            f"paras={record['paragraph_count']}  "
            f"dates={record['date_mention_count']}  "
            f"versions={record['version_mention_count']}"
        )

    print()
    print(f"输出: {OUTPUT_JSONL}")
    print(f"输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
