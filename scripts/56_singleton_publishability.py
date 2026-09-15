#!/usr/bin/env python3
"""
Step 16 - Singleton publishability gate
========================================

对 singleton_candidates_v1 的 858 条 candidate
做发布判定 (10 条/批, 86 批, glm-5.3-flash,
配置同 Step 14/15.4)。

三 verdict (同 20/28/50):

    publish / manual_review / reject

singleton 特有的判定维度
(在 15.4 KB-worthiness 之上更严):

- self_contained: 答案脱离聊天上下文是否可读
  (聊天抽取答案常见悬空指代: "刚才那个" /
   "上面说的" / "这个问题")
- client_specific_risk: 是否含客户特定信息
  (人名/店名/专属配置; 0029 "如康总" 教训)
- temporal_risk: 时效风险
- privacy_phone: deterministic 前置守卫,
  question/answer 含手机号模式 → 强制 manual_review

deterministic 守卫 (publish 降级):

- self_contained == no
- client_specific_risk == high
- privacy_phone 前置标记
- answer < 20 字

注意:

- resolution / issue_date 不自动拒绝
- 858 条是原始 Q/A 对, 答案来自 Step 7 抽取,
  本 gate 不改写任何内容
- publish 通过者进入 official KB v3 (Step 16.2)

断点续跑: 每批落盘, 重跑跳过已完成。

用法:

    .venv/bin/python scripts/56_singleton_publishability.py
    .venv/bin/python scripts/56_singleton_publishability.py --dry-run
    .venv/bin/python scripts/56_singleton_publishability.py --only-batch 0
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CANDIDATES_FILE = (
    OUTPUT_DIR / "singleton_candidates_v1.jsonl"
)

OUTPUT_JSONL = (
    OUTPUT_DIR / "singleton_publishability_v1.jsonl"
)
OUTPUT_XLSX = (
    OUTPUT_DIR / "singleton_publishability_v1.xlsx"
)
ERROR_LOG = (
    OUTPUT_DIR
    / "singleton_publishability_llm_errors.log"
)

MODEL = "glm-5.3-flash"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 180
BATCH_SIZE = 10
ANSWER_PREVIEW_CHARS = 600

PHONE_PATTERN = re.compile(
    r"(?<!\d)1[3-9]\d{9}(?!\d)"
)


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


# ============================================================
# deterministic 前置标记
# ============================================================

def pre_flags(record):

    flags = []

    text = (
        record["question"]
        + "\n"
        + record["answer"]
    )

    if PHONE_PATTERN.search(text):
        flags.append("PRIVACY_PHONE")

    if len(record["answer"]) < 20:
        flags.append("ANSWER_TOO_SHORT")

    return flags


# ============================================================
# prompt
# ============================================================

def build_item_block(items, start, end):

    lines = []

    for index in range(start, end):
        item = items[index]
        answer = item["answer"][
            :ANSWER_PREVIEW_CHARS
        ]

        pre = (
            "存在前置标记: "
            + ", ".join(item["pre_flags"])
            if item["pre_flags"]
            else "无"
        )

        lines.append(
            f"### item_index={index}\n"
            f"issue_date: {item['issue_date']}\n"
            f"resolution: {item['resolution']}\n"
            f"knowledge_value: "
            f"{item['knowledge_value']}\n"
            f"problem_type: {item['problem_type']}\n"
            f"{pre}\n"
            f"问题: {item['question']}\n"
            f"答案: {answer}"
        )

    return "\n\n".join(lines)


def build_prompt(items, start, end):

    return f"""你是一个严格的知识库发布审核员。

下面有 {end - start} 条候选知识条目
(从客服聊天抽取的问答对, 问题与答案已通过
准入筛选与去重)。对每条独立做发布判定。

## 三 verdict

publish / manual_review / reject

## publish 要求整体满足

1. self_contained = yes:
   答案脱离聊天上下文独立可读。
   聊天抽取答案常见悬空指代
   ("刚才那个"、"上面说的"、"这个问题"、
   只呼应会话里前文的表述),
   凡依赖上下文才能理解的 → 至多 manual_review
2. client_specific_risk = none 或 low:
   不含客户特定信息
   (具体人名 / 门店名 / 专属配置 /
   只对这一个客户成立的细节)。
   出现具体客户名或门店专名 → 至少 high
3. temporal_risk 低, 或答案本身是稳定机制
4. 答案内容完整、可直接指导操作或说明规则
5. 没有把单次事故写成通用规则

## 注意

- resolution=unresolved/partial 不自动拒绝
- issue_date 旧不自动拒绝
- 宁可 manual_review 不要勉强 publish
- reject 用于: 答非所问 / 纯个案 / 无实质知识 /
  完全依赖会话上下文

## 待判定条目

{build_item_block(items, start, end)}

## 输出 JSON 格式（不要输出 JSON 以外的任何文字）

{{
  "judgments": [
    {{
      "item_index": <对应 item_index>,
      "self_contained": "yes | partial | no",
      "client_specific_risk": "none | low | high",
      "temporal_risk": "none | low | high",
      "verdict": "publish | manual_review | reject",
      "reason": "一句话核心理由",
      "notes": ["备注, 可为空数组"]
    }}
  ]
}}"""


# ============================================================
# Schema 与 LLM 调用
# ============================================================

class SingletonGateJudgment(BaseModel):

    item_index: int
    self_contained: Literal[
        "yes", "partial", "no"
    ]
    client_specific_risk: Literal[
        "none", "low", "high"
    ]
    temporal_risk: Literal["none", "low", "high"]
    verdict: Literal[
        "publish", "manual_review", "reject"
    ]
    reason: str
    notes: List[str] = Field(default_factory=list)


class BatchGateJudgment(BaseModel):

    judgments: List[SingletonGateJudgment] = Field(
        min_length=1
    )


ENUM_VALUES = {
    "self_contained": ["yes", "partial", "no"],
    "client_specific_risk": ["none", "low", "high"],
    "temporal_risk": ["none", "low", "high"],
    "verdict": [
        "publish", "manual_review", "reject"
    ],
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
        "4. notes 是字符串数组;\n"
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
                BatchGateJudgment.model_validate(
                    data
                )
            )

            indexes = sorted(
                j.item_index
                for j in judgment.judgments
            )

            if indexes != expected:
                raise ValueError(
                    f"item_index 覆盖不完整: 期望 "
                    f"{expected}, 实际 {indexes}"
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

VERDICT_RANK = {
    "publish": 2,
    "manual_review": 1,
    "reject": 0,
}


def apply_guards(item, judged):

    downgrades = []
    verdict = judged.verdict

    if verdict != "publish":
        return verdict, downgrades

    if judged.self_contained == "no":
        downgrades.append(
            "GUARD_NOT_SELF_CONTAINED"
        )

    if judged.client_specific_risk == "high":
        downgrades.append(
            "GUARD_CLIENT_SPECIFIC_HIGH"
        )

    if "PRIVACY_PHONE" in item["pre_flags"]:
        downgrades.append("GUARD_PRIVACY_PHONE")

    if "ANSWER_TOO_SHORT" in item["pre_flags"]:
        downgrades.append("GUARD_ANSWER_TOO_SHORT")

    if downgrades:
        return "manual_review", downgrades

    return "publish", downgrades


# ============================================================
# 主流程
# ============================================================

def main():

    dry_run = "--dry-run" in sys.argv
    only_batch = None

    if "--only-batch" in sys.argv:
        only_batch = int(
            sys.argv[
                sys.argv.index("--only-batch") + 1
            ]
        )

    raw_items = load_jsonl(CANDIDATES_FILE)

    items = []

    for record in raw_items:
        items.append({
            "issue_key": record["issue_key"],
            "question": record["question"],
            "answer": record["answer"],
            "issue_date": record["issue_date"],
            "resolution": record["resolution"],
            "knowledge_value": record[
                "knowledge_value"
            ],
            "problem_type": record[
                "problem_type"
            ],
            "pre_flags": pre_flags(record),
        })

    flagged = sum(
        1 for item in items if item["pre_flags"]
    )

    batches = [
        (start, min(start + BATCH_SIZE, len(items)))
        for start in range(0, len(items), BATCH_SIZE)
    ]

    print(
        f"待审: {len(items)} 条, {len(batches)} 批, "
        f"前置标记 {flagged} 条"
    )

    if dry_run:

        phone = sum(
            1 for item in items
            if "PRIVACY_PHONE"
            in item["pre_flags"]
        )
        print(f"PRIVACY_PHONE: {phone}")

        sample = build_prompt(items, 0, 2)
        print(f"prompt 样例: {len(sample)} chars")
        print("dry-run 完成")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY 环境变量"
        )

    done = {}

    if OUTPUT_JSONL.exists():
        for record in load_jsonl(OUTPUT_JSONL):
            done[record["issue_key"]] = record

        print(f"断点续跑: 已有 {len(done)} 条")

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

            batch_keys = [
                items[i]["issue_key"]
                for i in range(start, end)
            ]

            if all(
                key in done for key in batch_keys
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

                verdict, downgrades = (
                    apply_guards(item, j)
                )

                record = {
                    "issue_key": item[
                        "issue_key"
                    ],
                    "question": item["question"],
                    "resolution": item[
                        "resolution"
                    ],
                    "pre_flags": item["pre_flags"],
                    "item_index": j.item_index,
                    "self_contained": (
                        j.self_contained
                    ),
                    "client_specific_risk": (
                        j.client_specific_risk
                    ),
                    "temporal_risk": (
                        j.temporal_risk
                    ),
                    "verdict_raw": j.verdict,
                    "verdict_final": verdict,
                    "guard_downgrades": downgrades,
                    "reason": j.reason,
                    "notes": j.notes,
                    "model": MODEL,
                    "attempt": judged["attempt"],
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
        "total": len(df),
        "publish": int(
            (df["verdict_final"] == "publish").sum()
        ),
        "manual_review": int(
            (df["verdict_final"]
             == "manual_review").sum()
        ),
        "reject": int(
            (df["verdict_final"] == "reject").sum()
        ),
        "guard_downgraded": int(
            df["guard_downgrades"].apply(bool).sum()
        ),
    }

    verdict_rows = df[
        [
            "issue_key",
            "question",
            "resolution",
            "pre_flags",
            "self_contained",
            "client_specific_risk",
            "temporal_risk",
            "verdict_raw",
            "verdict_final",
            "guard_downgrades",
            "reason",
        ]
    ]

    with pd.ExcelWriter(
        OUTPUT_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame([summary]).to_excel(
            writer, sheet_name="summary", index=False
        )
        verdict_rows.to_excel(
            writer, sheet_name="verdicts", index=False
        )
        df[
            df["verdict_final"] == "publish"
        ].to_excel(
            writer, sheet_name="publish", index=False
        )
        df[
            df["guard_downgrades"].apply(bool)
        ].to_excel(
            writer,
            sheet_name="guard_downgraded",
            index=False,
        )

    print()
    print("=" * 60)
    print("Singleton publishability gate 完成")
    print("=" * 60)
    print(
        f"publish={summary['publish']}  "
        f"manual_review={summary['manual_review']}  "
        f"reject={summary['reject']}"
    )
    print(f"守卫降级: {summary['guard_downgraded']}")
    print()
    print(f"输出: {OUTPUT_JSONL}")
    print(f"输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
