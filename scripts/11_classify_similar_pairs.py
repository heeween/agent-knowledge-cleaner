import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

INPUT_FILE = (
    OUTPUT_DIR
    / "similar_question_pairs.xlsx"
)

OUTPUT_JSONL = (
    OUTPUT_DIR
    / "pair_classifications.jsonl"
)

OUTPUT_XLSX = (
    OUTPUT_DIR
    / "pair_classifications.xlsx"
)

MODEL = "qwen3.5-flash"

MIN_SIMILARITY = 0.88

MAX_WORKERS = 5
MAX_RETRIES = 4
REQUEST_TIMEOUT = 120


# ============================================================
# Schema
# ============================================================

class PairDecision(BaseModel):

    relation: Literal[
        "same_intent",
        "parent_child",
        "related",
        "different",
    ]

    confidence: float

    reason: str

    canonical_question: str | None = None


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
你正在整理 CRM 客服知识库。

现在有两个从客服聊天中抽取出来的
question_normalized。

你的任务不是回答问题，
而是判断两个问题之间的知识关系。

必须严格使用以下四种 relation：

1. same_intent

两个问题本质上在问同一个 CRM 知识点，
正常情况下应该由同一条知识答案回答。

允许：
- 不同措辞
- 同义表达
- 轻微上下文差异
- 一个问题比另一个表达更完整

例如：

A：如何重置 CRM 账号密码？
B：CRM 账号密码忘记后如何重置？

relation = same_intent


2. parent_child

两个问题属于同一知识主题，
但一个问题明显更宽泛，
另一个问题只是其中一个具体子问题。

例如：

A：CRM 系统无法登录时如何处理？
B：CRM 登录失败时如何重置密码？

密码重置只是“无法登录”的一种处理方式，
不能直接认为两个问题完全相同。

relation = parent_child


3. related

两个问题讨论同一功能、模块或现象，
但用户真正希望得到的答案不同，
不能合并成一条知识问题。

例如：

A：预计保养里程是什么意思？
B：预计保养里程为什么出现异常？

relation = related

再例如：

A：短信发送失败的原因是什么？
B：短信发送失败是否会扣费？

relation = related


4. different

虽然 embedding 相似，
但实际上属于不同知识问题，
甚至只是因为包含相同 CRM 词汇而相似。


非常重要：

- 判断的核心是：
  “这两个问题是否可以由同一个知识答案直接回答？”

- 不要因为属于同一个 crm_module 就判断 same_intent。

- 不要因为出现相同关键词就判断 same_intent。

- 不要因为两个问题的答案恰好相似就判断 same_intent。

- “为什么”“怎么处理”“在哪里查看”
  如果用户实际需要的信息不同，
  通常不是 same_intent。

- 一个问题询问“是否支持”，
  另一个问题询问“如何操作”，
  通常不是 same_intent，
  除非两者本质上确实指向同一个产品规则。

- 一个问题询问功能定义，
  另一个问题询问异常原因，
  不是 same_intent。

- 一个宽泛问题与一个具体故障原因，
  优先考虑 parent_child，而不是 same_intent。


canonical_question：

只有 relation = same_intent 时填写。

请根据两个问题生成一个：
- 独立
- 简洁
- 可搜索
- 不包含具体客户/门店/姓名
- 保留核心 CRM 业务含义

的标准问题。

如果 relation 不是 same_intent，
canonical_question 必须为 null。

confidence 表示你对关系判断的把握程度，
范围 0 到 1。

reason 用一句简短的话解释判断依据。

输出要求：

你必须只输出一个合法的 JSON object。
不要输出 Markdown。
不要输出 ```json 代码块。
不要输出 JSON 之外的任何解释文字。

JSON 格式必须严格为：

{
  "relation": "same_intent | parent_child | related | different",
  "confidence": 0.0,
  "reason": "简短判断理由",
  "canonical_question": "标准问题或 null"
}

其中：
- confidence 必须是 0 到 1 之间的数字。
- relation 只能使用规定的四种值。
- relation 不是 same_intent 时，canonical_question 必须为 null。
"""


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
# 工具
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


def make_pair_key(
    issue_key_a,
    issue_key_b,
):

    keys = sorted(
        [
            str(issue_key_a),
            str(issue_key_b),
        ]
    )

    return (
        keys[0]
        + "||"
        + keys[1]
    )


def load_processed():

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

                pair_key = obj.get(
                    "pair_key"
                )

                if pair_key:
                    processed.add(
                        pair_key
                    )

            except Exception:
                continue

    return processed


# ============================================================
# 请求
# ============================================================

def classify_pair(row):

    client = build_client()

    pair_key = make_pair_key(
        row["issue_key_a"],
        row["issue_key_b"],
    )

    user_prompt = f"""
请判断下面两个 CRM 知识问题的关系。

Question A:
{clean_text(row["question_normalized_a"])}

Question B:
{clean_text(row["question_normalized_b"])}

辅助信息：

CRM module A:
{clean_text(row.get("crm_module_a"))}

CRM module B:
{clean_text(row.get("crm_module_b"))}

Embedding cosine similarity:
{float(row["similarity"]):.6f}

注意：
辅助信息只能帮助理解，
不能仅凭模块或 similarity
判断为 same_intent。
"""

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            started = time.time()

            response = (
                client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content":
                                user_prompt,
                        },
                    ],
                    response_format={
                        "type": "json_object"
                    },
                    temperature=0,
                    extra_body={
                        "enable_thinking":
                            False
                    },
                )
            )

            elapsed = (
                time.time()
                - started
            )

            content = (
                response
                .choices[0]
                .message
                .content
            )

            data = json.loads(
                content
            )

            decision = (
                PairDecision
                .model_validate(
                    data
                )
            )

            # 非 same_intent 不允许 canonical
            if (
                decision.relation
                != "same_intent"
            ):
                decision.canonical_question = None

            return {
                "pair_key":
                    pair_key,

                "issue_key_a":
                    row[
                        "issue_key_a"
                    ],

                "issue_key_b":
                    row[
                        "issue_key_b"
                    ],

                "similarity":
                    float(
                        row["similarity"]
                    ),

                "question_a":
                    clean_text(
                        row[
                            "question_normalized_a"
                        ]
                    ),

                "question_b":
                    clean_text(
                        row[
                            "question_normalized_b"
                        ]
                    ),

                "crm_module_a":
                    clean_text(
                        row.get(
                            "crm_module_a"
                        )
                    ),

                "crm_module_b":
                    clean_text(
                        row.get(
                            "crm_module_b"
                        )
                    ),

                "relation":
                    decision.relation,

                "confidence":
                    decision.confidence,

                "canonical_question":
                    (
                        decision
                        .canonical_question
                    ),

                "reason":
                    decision.reason,

                "request_seconds":
                    elapsed,
            }

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                time.sleep(
                    2 ** attempt
                )

    raise last_error


# ============================================================
# 导出 Excel
# ============================================================

def export_excel():

    rows = []

    if OUTPUT_JSONL.exists():

        with OUTPUT_JSONL.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:
                    rows.append(
                        json.loads(line)
                    )
                except Exception:
                    pass

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        return

    relation_stats = (
        df["relation"]
        .value_counts()
        .rename_axis(
            "relation"
        )
        .reset_index(
            name="count"
        )
    )

    confidence_stats = (
        df.groupby(
            "relation",
            dropna=False,
        )["confidence"]
        .agg(
            [
                "count",
                "mean",
                "min",
                "max",
            ]
        )
        .reset_index()
    )

    same_intent = df[
        df["relation"]
        == "same_intent"
    ].copy()

    review = df[
        (
            df["confidence"]
            < 0.85
        )
        |
        (
            (
                df["similarity"]
                >= 0.95
            )
            &
            (
                df["relation"]
                != "same_intent"
            )
        )
    ].copy()

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "classified_pair_count",
                "value":
                    len(df),
            },
            {
                "metric":
                    "same_intent_count",
                "value":
                    len(same_intent),
            },
            {
                "metric":
                    "review_count",
                "value":
                    len(review),
            },
            {
                "metric":
                    "min_similarity",
                "value":
                    MIN_SIMILARITY,
            },
        ]
    )

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl",
    ) as writer:

        summary.to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )

        relation_stats.to_excel(
            writer,
            sheet_name="relation_stats",
            index=False,
        )

        confidence_stats.to_excel(
            writer,
            sheet_name="confidence_stats",
            index=False,
        )

        df.to_excel(
            writer,
            sheet_name="all_pairs",
            index=False,
        )

        same_intent.to_excel(
            writer,
            sheet_name="same_intent",
            index=False,
        )

        review.to_excel(
            writer,
            sheet_name="review",
            index=False,
        )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print(
        "Step 8.4 - LLM classify similar pairs"
    )
    print("=" * 70)

    df = pd.read_excel(
        INPUT_FILE,
        sheet_name="all_pairs",
    )

    df = df[
        df["similarity"]
        >= MIN_SIMILARITY
    ].copy()

    print(
        f"Similarity >= "
        f"{MIN_SIMILARITY}: "
        f"{len(df)} pairs"
    )

    processed = load_processed()

    print(
        f"已处理: "
        f"{len(processed)}"
    )

    tasks = []

    for _, row in df.iterrows():

        pair_key = make_pair_key(
            row["issue_key_a"],
            row["issue_key_b"],
        )

        if pair_key in processed:
            continue

        tasks.append(
            row
        )

    print(
        f"本次待处理: "
        f"{len(tasks)}"
    )

    if not tasks:

        export_excel()

        print(
            "没有待处理 pair。"
        )

        return

    lock = threading.Lock()

    completed = 0
    total = len(tasks)

    started_at = time.time()

    with OUTPUT_JSONL.open(
        "a",
        encoding="utf-8",
    ) as fout:

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            future_map = {
                executor.submit(
                    classify_pair,
                    row,
                ): row
                for row in tasks
            }

            for future in as_completed(
                future_map
            ):

                row = future_map[
                    future
                ]

                try:

                    result = (
                        future.result()
                    )

                except Exception as exc:

                    print()
                    print(
                        "[FAILED]"
                    )

                    print(
                        row[
                            "issue_key_a"
                        ],
                        row[
                            "issue_key_b"
                        ],
                    )

                    print(
                        type(exc).__name__,
                        exc,
                    )

                    continue

                with lock:

                    fout.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    fout.flush()

                    completed += 1

                elapsed = (
                    time.time()
                    - started_at
                )

                avg = (
                    elapsed
                    / completed
                )

                eta = (
                    avg
                    * (
                        total
                        - completed
                    )
                )

                if (
                    completed % 20 == 0
                    or completed == total
                ):

                    print(
                        f"[{completed}/"
                        f"{total}] "
                        f"{completed/total*100:.1f}% "
                        f"| ETA "
                        f"{eta/60:.1f} min"
                    )

    export_excel()

    print()
    print("=" * 70)
    print("完成")
    print("=" * 70)

    print(
        f"输出 JSONL: "
        f"{OUTPUT_JSONL}"
    )

    print(
        f"输出 Excel: "
        f"{OUTPUT_XLSX}"
    )


if __name__ == "__main__":
    main()