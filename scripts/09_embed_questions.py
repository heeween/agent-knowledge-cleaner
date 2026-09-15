import os
import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "extracted_issues.xlsx"

OUTPUT_JSONL = OUTPUT_DIR / "question_embeddings.jsonl"

MODEL = "text-embedding-v4"

# 官方推荐的通用维度
EMBEDDING_DIMENSIONS = 1024

# text-embedding-v4 单批最多 10 条
BATCH_SIZE = 10

MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Client
# ============================================================

def build_client():
    return OpenAI(
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


# ============================================================
# 工具函数
# ============================================================

def clean_text(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def load_processed_keys():
    processed = set()

    if not OUTPUT_JSONL.exists():
        return processed

    with OUTPUT_JSONL.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

                issue_key = obj.get(
                    "issue_key"
                )

                if issue_key:
                    processed.add(
                        issue_key
                    )

            except Exception:
                continue

    return processed


def make_issue_key(row, row_index):
    source_candidate_id = clean_text(
        row.get("source_candidate_id")
    )

    issue_seq = clean_text(
        row.get("issue_seq")
    )

    if (
        source_candidate_id
        and issue_seq
    ):
        return (
            f"{source_candidate_id}"
            f"#{issue_seq}"
        )

    if source_candidate_id:
        return (
            f"{source_candidate_id}"
            f"#ROW-{row_index + 1:05d}"
        )

    return (
        f"ROW-{row_index + 1:05d}"
    )


# ============================================================
# API 请求
# ============================================================

def embed_batch(
    client,
    texts,
):
    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            start = time.time()

            response = (
                client.embeddings.create(
                    model=MODEL,
                    input=texts,
                    dimensions=(
                        EMBEDDING_DIMENSIONS
                    ),
                )
            )

            elapsed = (
                time.time() - start
            )

            embeddings = [
                item.embedding
                for item in response.data
            ]

            if len(embeddings) != len(texts):
                raise RuntimeError(
                    "Embedding 返回数量与输入数量不一致："
                    f"{len(texts)} -> "
                    f"{len(embeddings)}"
                )

            return (
                embeddings,
                elapsed,
            )

        except Exception as exc:
            last_error = exc

            print()
            print(
                f"[retry {attempt}/"
                f"{MAX_RETRIES}] "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            if attempt < MAX_RETRIES:
                sleep_seconds = (
                    2 ** attempt
                )

                print(
                    f"等待 "
                    f"{sleep_seconds}s "
                    f"后重试..."
                )

                time.sleep(
                    sleep_seconds
                )

    raise last_error


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 70)
    print(
        "Step 8.2 - question_normalized embeddings"
    )
    print("=" * 70)

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name="extracted_issues",
    )

    print(
        f"读取 issue 数: {len(df)}"
    )

    if "question_normalized" not in df.columns:
        raise RuntimeError(
            "找不到 question_normalized 字段"
        )

    processed = load_processed_keys()

    print(
        f"已生成 embedding: "
        f"{len(processed)}"
    )

    # --------------------------------------------------------
    # 构造任务
    # --------------------------------------------------------

    tasks = []

    for idx, row in df.iterrows():

        question = clean_text(
            row.get(
                "question_normalized"
            )
        )

        if not question:
            continue

        issue_key = make_issue_key(
            row,
            idx,
        )

        if issue_key in processed:
            continue

        tasks.append(
            {
                "row_index": idx,
                "issue_key": issue_key,
                "source_candidate_id":
                    clean_text(
                        row.get(
                            "source_candidate_id"
                        )
                    ),
                "issue_seq":
                    clean_text(
                        row.get(
                            "issue_seq"
                        )
                    ),
                "question_normalized":
                    question,
                "crm_module":
                    clean_text(
                        row.get(
                            "crm_module"
                        )
                    ),
            }
        )

    print(
        f"本次待处理: {len(tasks)}"
    )

    if not tasks:
        print(
            "没有需要处理的数据。"
        )
        return

    client = build_client()

    print()
    print(
        f"Embedding model: {MODEL}"
    )
    print(
        f"Dimensions: "
        f"{EMBEDDING_DIMENSIONS}"
    )
    print(
        f"Batch size: {BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # 先测试第一批
    # --------------------------------------------------------

    test_batch = tasks[
        :min(
            BATCH_SIZE,
            len(tasks),
        )
    ]

    test_texts = [
        x["question_normalized"]
        for x in test_batch
    ]

    print()
    print(
        "正在执行首批 API 测试..."
    )

    try:
        _, test_elapsed = embed_batch(
            client,
            test_texts,
        )

        print(
            "首批测试成功："
            f"{len(test_texts)} 条 / "
            f"{test_elapsed:.2f}s"
        )

    except Exception as exc:
        print()
        print("=" * 70)
        print("Embedding API 测试失败")
        print("=" * 70)
        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )
        print()
        print(
            "如果错误提示模型不存在或 "
            "embeddings 接口不支持，"
            "先不要改其他代码，把这个错误发给我。"
        )
        return

    # --------------------------------------------------------
    # 正式处理
    # --------------------------------------------------------

    total = len(tasks)
    completed = 0

    started_at = time.time()

    with OUTPUT_JSONL.open(
        "a",
        encoding="utf-8",
    ) as fout:

        for batch_start in range(
            0,
            total,
            BATCH_SIZE,
        ):
            batch = tasks[
                batch_start:
                batch_start + BATCH_SIZE
            ]

            texts = [
                item[
                    "question_normalized"
                ]
                for item in batch
            ]

            try:
                embeddings, elapsed = (
                    embed_batch(
                        client,
                        texts,
                    )
                )

            except Exception as exc:
                print()
                print(
                    "批次失败，停止运行。"
                )
                print(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
                print(
                    "已经写入的结果会保留，"
                    "下次可以 resume。"
                )
                raise

            for item, vector in zip(
                batch,
                embeddings,
            ):
                record = {
                    "issue_key":
                        item["issue_key"],

                    "source_candidate_id":
                        item[
                            "source_candidate_id"
                        ],

                    "issue_seq":
                        item["issue_seq"],

                    "question_normalized":
                        item[
                            "question_normalized"
                        ],

                    "crm_module":
                        item["crm_module"],

                    "embedding_model":
                        MODEL,

                    "embedding_dimensions":
                        len(vector),

                    "embedding":
                        vector,
                }

                fout.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            fout.flush()

            completed += len(batch)

            total_elapsed = (
                time.time()
                - started_at
            )

            avg_seconds = (
                total_elapsed
                / completed
            )

            remaining = (
                total - completed
            )

            eta_seconds = (
                remaining
                * avg_seconds
            )

            percent = (
                completed
                / total
                * 100
            )

            print(
                f"[{completed:4d}/"
                f"{total}] "
                f"{percent:6.2f}% | "
                f"batch {elapsed:5.2f}s | "
                f"ETA "
                f"{eta_seconds / 60:6.1f} min"
            )

    print()
    print("=" * 70)
    print("Embedding 完成")
    print("=" * 70)

    print(
        f"本次生成: {completed}"
    )

    print(
        f"输出文件: {OUTPUT_JSONL}"
    )


if __name__ == "__main__":
    main()