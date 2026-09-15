from pathlib import Path
import re
import json


RAW_DIR = Path("data/raw/markdowns")
OUTPUT_FILE = Path("output/messages.jsonl")


MESSAGE_PATTERN = re.compile(
    r"^【(?P<timestamp>[^】]+)】(?P<speaker>[^:：]+)[:：]\s*(?P<content>.*)$"
)


def parse_file(file_path: Path, document_id: str):

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    messages = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        match = MESSAGE_PATTERN.match(line)

        if not match:
            continue

        messages.append({
            "document_id": document_id,
            "line_number": line_number,
            "timestamp": match.group("timestamp"),
            "speaker": match.group("speaker").strip(),
            "content": match.group("content").strip(),
        })

    return messages


def main():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        RAW_DIR.glob("*.md")
    )

    total_messages = 0
    parsed_documents = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        for index, file_path in enumerate(
            files,
            start=1
        ):

            document_id = f"D{index:04d}"

            messages = parse_file(
                file_path,
                document_id
            )

            if messages:
                parsed_documents += 1

            for message in messages:

                message["filename"] = file_path.name

                output.write(
                    json.dumps(
                        message,
                        ensure_ascii=False
                    ) + "\n"
                )

                total_messages += 1

            print(
                f"[{index}/{len(files)}] "
                f"{file_path.name} "
                f"→ {len(messages)} 条消息"
            )

    print()
    print("=" * 60)
    print("聊天记录解析完成")
    print("=" * 60)
    print(f"Markdown 文件：{len(files)}")
    print(f"包含聊天消息的文件：{parsed_documents}")
    print(f"消息总数：{total_messages}")
    print(f"输出：{OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()