from pathlib import Path
import hashlib
import pandas as pd
import re


RAW_DIR = Path("data/raw/markdowns")
OUTPUT_FILE = Path("output/document_inventory.xlsx")


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def extract_title(content: str, filename: str) -> str:
    # 优先使用 Markdown 一级标题
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)

    if match:
        return match.group(1).strip()

    # 没有一级标题，就使用文件名
    return filename


def count_words(content: str) -> int:
    return len(re.findall(r"\S+", content))


def analyze_file(file_path: Path):
    content = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    stat = file_path.stat()

    return {
        "filename": file_path.name,
        "relative_path": str(file_path.relative_to(RAW_DIR)),
        "file_size": stat.st_size,
        "modified_time": pd.to_datetime(stat.st_mtime, unit="s"),
        "sha256": calculate_sha256(file_path),
        "title": extract_title(content, file_path.name),
        "word_count": count_words(content),
    }


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"目录不存在: {RAW_DIR}"
        )

    files = sorted(RAW_DIR.rglob("*.md"))

    print(f"发现 Markdown 文件: {len(files)}")

    documents = []

    for index, file_path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {file_path.name}")

        try:
            documents.append(
                analyze_file(file_path)
            )
        except Exception as e:
            print(f"处理失败: {file_path}")
            print(f"错误: {e}")

    df = pd.DataFrame(documents)

    # 增加 document_id
    df.insert(
        0,
        "document_id",
        [f"D{i:04d}" for i in range(1, len(df) + 1)]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("文档盘点完成")
    print("=" * 60)
    print(f"文档数量: {len(df)}")
    print(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()