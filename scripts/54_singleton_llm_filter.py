#!/usr/bin/env python3
"""
Step 15.4 - Singleton funnel Layer D (LLM KB-worthiness)
=========================================================

对 C 层去重后的幸存者做 LLM 批量判定
(10 条/批, glm-5.3-flash, 配置同 Step 14 gate)。

每条判定维度:

- qa_aligned: answer 是否回答 question (yes/partial/no)
- reusable: 通用可复用知识 vs 单次个案 (yes/partial/no)
- incident_risk: 把单次事故写成通用规则的风险
- stable_limitation: 是否"当前不支持/限制X"型
  稳定产品知识 (feature_request 红线字段)
- kb_worthy: candidate / reject

deterministic 守卫:

- kb_worthy=candidate 要求 qa_aligned != no
- resolution=feature_request 的 candidate
  要求 stable_limitation=true
  (否则 GUARD_FEATURE_REQUEST_NOT_STABLE 强制 reject)

红线:

- 不要求 resolution == resolved,
  unresolved/partial 按内容正常判断
- 只判定 KB 价值, 不改写任何内容

断点续跑:

- 每批完成即追加写 output jsonl
- 重跑自动跳过已判定 issue_key

用法:

    .venv/bin/python scripts/54_singleton_llm_filter.py
    .venv/bin/python scripts/54_singleton_llm_filter.py --dry-run
    .venv/bin/python scripts/54_singleton_llm_filter.py --only-batch 0
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

DEDUP_FILE = OUTPUT_DIR / "singleton_dedup_v1.jsonl"
SURVIVORS_FILE = (
    OUTPUT_DIR / "singleton_survivors_ab_v1.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR / "singleton_llm_filter_v1.jsonl"
)
OUTPUT_XLSX = (
    OUTPUT_DIR / "singleton_llm_filter_v1.xlsx"
)
ERROR_LOG = (
    OUTPUT_DIR
    / "singleton_llm_filter_llm_errors.log"
)

MODEL = "glm-5.3-flash"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 180
BATCH_SIZE = 10
ANSWER_PREVIEW_CHARS = 600


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


# ============================================================
# 输入组装
# ============================================================

def load_items():

    dedup = load_jsonl(DEDUP_FILE)
    survivors = {
        record["issue_key"]: record
        for record in load_jsonl(SURVIVORS_FILE)
    }

    kept = [
        record for record in dedup
        if record["decision"] == "kept"
    ]

    items = []

    for record in kept:
        survivor = survivors.get(
            record["issue_key"]
        )

        if survivor is None:
            raise KeyError(
                f"{record['issue_key']} "
                f"在 survivors 文件中缺失"
            )

        answer = survivor.get("answer") or ""

        # 守卫: 答案必须非空
        # (dedup 决策文件不携带 answer,
        #  必须从 survivors 文件 join)
        if not answer.strip():
            raise RuntimeError(
                f"{record['issue_key']} 答案为空"
            )

        items.append({
            "issue_key": record["issue_key"],
            "question": survivor["question"],
            "answer": answer,
            "issue_date": survivor["issue_date"],
            "confidence": survivor["confidence"],
            "knowledge_value": survivor[
                "knowledge_value"
            ],
            "resolution": survivor["resolution"],
        })

    return items


def build_item_block(items, start, end):

    lines = []

    for index in range(start, end):
        item = items[index]
        answer = item["answer"][
            :ANSWER_PREVIEW_CHARS
        ]

        lines.append(
            f"### item_index={index}\n"
            f"issue_key: {item['issue_key']}\n"
            f"issue_date: {item['issue_date']}\n"
            f"resolution: {item['resolution']}\n"
            f"knowledge_value: "
            f"{item['knowledge_value']}\n"
            f"问题: {item['question']}\n"
            f"答案: {answer}"
        )

    return "\n\n".join(lines)


def build_prompt(items, start, end):

    return f"""你是一个严格的知识库准入审核员。

下面有 {end - start} 条从客服聊天中抽取的问答
(singleton issue, 已通过质量硬过滤与去重)。
对每条独立判定是否值得进入知识库候选
(kb_worthy = candidate / reject)。

## 判定维度

1. qa_aligned: 答案是否回答了问题
   - yes: 答案直接回答问题
   - partial: 答案只部分回应或答到相邻问题
   - no: 答案与问题无关 / 没有实质回答
2. reusable: 内容是否为通用可复用知识
   - yes: 任何门店/用户遇到同样问题都适用
   - partial: 大体适用但有场景限定
   - no: 只对这一次会话/这一个客户成立
3. incident_risk: 把单次事故写成通用规则的风险
   - high: 内容明显是一次性故障/个案处理
   - low: 有场景限定但机制本身通用
   - none: 纯产品机制/操作流程
4. stable_limitation: 答案是否为
   "当前不支持/无法实现/存在限制X"型稳定产品知识
   (该类知识有价值, 即使 resolution 不是 resolved)
5. kb_worthy: 最终判定
   - candidate: qa_aligned 不是 no,
     且内容具备可复用价值
   - reject: 答非所问 / 纯个案 / 无实质知识

## 注意

- resolution=unresolved/partial 不自动 reject
- resolution=feature_request 只有在
  stable_limitation=true 时才允许 candidate
- issue_date 仅作背景参考, 不因日期旧而 reject
- 宁可漏选不可误收

## 待判定条目

{build_item_block(items, start, end)}

## 输出 JSON 格式（不要输出 JSON 以外的任何文字）

{{
  "judgments": [
    {{
      "item_index": <对应 item_index>,
      "qa_aligned": "yes | partial | no",
      "reusable": "yes | partial | no",
      "incident_risk": "none | low | high",
      "stable_limitation": true | false,
      "kb_worthy": "candidate | reject",
      "reason": "一句话核心理由"
    }}
  ]
}}"""


# ============================================================
# Schema
# ============================================================

class SingletonJudgment(BaseModel):

    item_index: int
    qa_aligned: Literal["yes", "partial", "no"]
    reusable: Literal["yes", "partial", "no"]
    incident_risk: Literal["none", "low", "high"]
    stable_limitation: bool
    kb_worthy: Literal["candidate", "reject"]
    reason: str


class BatchJudgment(BaseModel):

    judgments: List[SingletonJudgment] = Field(
        min_length=1
    )


ENUM_VALUES = {
    "qa_aligned": ["yes", "partial", "no"],
    "reusable": ["yes", "partial", "no"],
    "incident_risk": ["none", "low", "high"],
    "kb_worthy": ["candidate", "reject"],
}


def build_client():

    return OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
        max_retries=0,
    )


def strip_json_fences(raw):

    text = raw.strip()

    if text.startswith("```"):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return text.strip()


def log_llm_error(batch_id, raw, error):

    with ERROR_LOG.open("a", encoding="utf-8") as f:

        f.write(
            "=" * 70
            + "\n"
            + f"[{datetime.now().isoformat()}] "
            + f"{batch_id}\n"
            + f"ERROR: {error}\n"
            + "RAW RESPONSE:\n"
            + raw
            + "\n"
        )


def build_repair_note(error, expected_indexes):

    enum_hint = "; ".join(
        f"{field}: {'/'.join(values)}"
        for field, values in ENUM_VALUES.items()
    )

    return (
        "你上一次的输出未通过 schema 校验，错误如下:\n"
        f"{error}\n\n"
        "请重新输出完整 JSON，要求:\n"
        "1. judgments 必须恰好覆盖以下 item_index:\n"
        f"{expected_indexes}\n"
        "2. 字段名与类型与要求完全一致;\n"
        f"3. 枚举取值严格使用 ({enum_hint});\n"
        "4. stable_limitation 是布尔值 true/false;\n"
        "5. 不要输出 JSON 以外的任何文字。"
    )


def judge_batch(client, items, start, end):

    prompt = build_prompt(items, start, end)
    expected = list(range(start, end))

    last_error = None
    repair_note = None
    last_raw = None

    for attempt in range(1, MAX_RETRIES + 1):

        started = time.time()

        messages = [
            {"role": "user", "content": prompt}
        ]

        if repair_note and last_raw:

            messages.append(
                {
                    "role": "assistant",
                    "content": last_raw,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": repair_note,
                }
            )

        response = (
            client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format={
                    "type": "json_object"
                },
                temperature=0,
                extra_body={
                    "reasoning_effort": "low"
                },
            )
        )

        elapsed = time.time() - started

        raw = strip_json_fences(
            response.choices[0].message.content
        )

        last_raw = raw

        try:

            data = json.loads(raw)

            judgment = (
                BatchJudgment.model_validate(data)
            )

            indexes = sorted(
                j.item_index
                for j in judgment.judgments
            )

            if indexes != expected:
                raise ValueError(
                    f"item_index 覆盖不完整: "
                    f"期望 {expected}, 实际 {indexes}"
                )

            if attempt > 1:
                log_llm_error(
                    f"batch_{start}_{end}",
                    raw,
                    f"RECOVERED_ON_ATTEMPT_{attempt}",
                )

            return {
                "judgment": judgment,
                "attempt": attempt,
                "request_seconds": round(
                    elapsed, 2
                ),
                "raw": raw,
            }

        except Exception as error:

            last_error = error
            log_llm_error(
                f"batch_{start}_{end}", raw, error
            )

            repair_note = build_repair_note(
                error, expected
            )

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"batch {start}-{end} 连续 "
        f"{MAX_RETRIES} 次失败: {last_error}"
    )


# ============================================================
# deterministic 守卫
# ============================================================

def apply_guards(item, judged):

    guards = []

    kb_worthy = judged.kb_worthy

    if (
        kb_worthy == "candidate"
        and judged.qa_aligned == "no"
    ):
        kb_worthy = "reject"
        guards.append(
            "GUARD_CANDIDATE_WITH_QA_ALIGNED_NO"
        )

    if (
        kb_worthy == "candidate"
        and item["resolution"]
        == "feature_request"
        and not judged.stable_limitation
    ):
        kb_worthy = "reject"
        guards.append(
            "GUARD_FEATURE_REQUEST_NOT_STABLE"
        )

    return kb_worthy, guards


def main():

    dry_run = "--dry-run" in sys.argv
    only_batch = None

    if "--only-batch" in sys.argv:
        only_batch = int(
            sys.argv[
                sys.argv.index("--only-batch") + 1
            ]
        )

    items = load_items()

    batches = [
        (start, min(start + BATCH_SIZE, len(items)))
        for start in range(
            0, len(items), BATCH_SIZE
        )
    ]

    print(f"待判定: {len(items)} 条, "
          f"{len(batches)} 批")

    if dry_run:
        sample = build_prompt(items, 0, 3)
        print(
            f"prompt 样例 (3 条): "
            f"{len(sample)} chars"
        )
        print("dry-run 完成")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY 环境变量"
        )

    # 断点续跑: 跳过已判定 issue_key
    done = {}

    if OUTPUT_JSONL.exists():
        for record in load_jsonl(OUTPUT_JSONL):
            done[record["issue_key"]] = record

        print(f"断点续跑: 已有 {len(done)} 条结果")

    client = build_client()

    output_records = list(done.values())

    with open(
        OUTPUT_JSONL, "a", encoding="utf-8"
    ) as output:

        for batch_index, (start, end) in (
            enumerate(batches)
        ):

            if only_batch is not None and (
                batch_index != only_batch
            ):
                continue

            batch_items = [
                item["issue_key"]
                for item in items[start:end]
            ]

            if all(
                key in done for key in batch_items
            ):
                continue

            print(
                f"[batch {batch_index}] "
                f"{start}-{end} 判定中...",
                flush=True,
            )

            judged = judge_batch(
                client, items, start, end
            )

            for j in judged["judgment"].judgments:
                item = items[j.item_index]

                kb_worthy, guards = apply_guards(
                    item, j
                )

                record = {
                    "issue_key": item[
                        "issue_key"
                    ],
                    "question": item["question"],
                    "answer_chars": len(
                        item["answer"]
                    ),
                    "knowledge_value": item[
                        "knowledge_value"
                    ],
                    "resolution": item[
                        "resolution"
                    ],
                    "issue_date": item[
                        "issue_date"
                    ],
                    "item_index": j.item_index,
                    "qa_aligned": j.qa_aligned,
                    "reusable": j.reusable,
                    "incident_risk": (
                        j.incident_risk
                    ),
                    "stable_limitation": (
                        j.stable_limitation
                    ),
                    "kb_worthy_raw": j.kb_worthy,
                    "kb_worthy_final": kb_worthy,
                    "guard_flags": guards,
                    "reason": j.reason,
                    "model": MODEL,
                    "attempt": judged[
                        "attempt"
                    ],
                }

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_records.append(record)

            output.flush()

    # 按 issue_key 去重保留最新
    # (批次中途失败可能留下部分旧记录)
    deduped = {}

    for record in output_records:
        deduped[record["issue_key"]] = record

    output_records = list(deduped.values())

    with open(
        OUTPUT_JSONL, "w", encoding="utf-8"
    ) as output:
        for record in output_records:
            output.write(
                json.dumps(
                    record, ensure_ascii=False
                )
                + "\n"
            )

    df = pd.DataFrame(output_records)

    summary = {
        "total_judged": len(df),
        "candidate": int(
            (df["kb_worthy_final"] == "candidate").sum()
        ),
        "reject": int(
            (df["kb_worthy_final"] == "reject").sum()
        ),
        "guard_flagged": int(
            df["guard_flags"].apply(bool).sum()
        ),
        "unresolved_partial_candidate": int(
            (
                (df["kb_worthy_final"] == "candidate")
                & (
                    df["resolution"].isin(
                        ["unresolved", "partial"]
                    )
                )
            ).sum()
        ),
    }

    with pd.ExcelWriter(
        OUTPUT_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame([summary]).to_excel(
            writer, sheet_name="summary", index=False
        )
        df.to_excel(
            writer, sheet_name="judgments", index=False
        )
        df[
            df["kb_worthy_final"] == "candidate"
        ].to_excel(
            writer,
            sheet_name="candidates",
            index=False,
        )
        df[
            df["guard_flags"].apply(bool)
        ].to_excel(
            writer,
            sheet_name="guard_flagged",
            index=False,
        )

    print()
    print("=" * 60)
    print("Singleton LLM 判定完成（Step 15.4）")
    print("=" * 60)
    print(f"判定: {summary['total_judged']}")
    print(f"candidate: {summary['candidate']}")
    print(f"reject: {summary['reject']}")
    print(f"守卫标记: {summary['guard_flagged']}")
    print(
        f"candidate 中 unresolved/partial: "
        f"{summary['unresolved_partial_candidate']}"
    )
    print()
    print(f"输出: {OUTPUT_JSONL}")
    print(f"输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
