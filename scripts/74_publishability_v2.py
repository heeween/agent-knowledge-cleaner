#!/usr/bin/env python3
"""
Step 18.6 - funnel v2 publishability gate
==========================================

与 scripts/56 同 prompt、同批量、同三 verdict,
并对 candidate 补充 deterministic 前置标记
(PRIVACY_PHONE / ANSWER_TOO_SHORT, 与 56 口径一致)。

    input:  output/funnel_v2_llm_filter.jsonl (candidate)
    output: output/funnel_v2_publishability.jsonl (断点续跑)

publish 条目即最终可导出为 KB-0646+ 的新知识。
"""

import concurrent.futures
import json
import os
import re
import time
from pathlib import Path

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

DEDUP_FILE = OUTPUT_DIR / "funnel_v2_dedup_survivors.jsonl"
FILTER_FILE = OUTPUT_DIR / "funnel_v2_llm_filter.jsonl"
GATE_FILE = OUTPUT_DIR / "funnel_v2_publishability.jsonl"

MODEL = "glm-5.3-flash"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
BATCH_SIZE = 10
ANSWER_PREVIEW_CHARS = 600
MAX_RETRIES = 4
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

ENUMS = {
    "self_contained": ["yes", "partial", "no"],
    "client_specific_risk": ["none", "low", "high"],
    "temporal_risk": ["none", "low", "high"],
    "verdict": ["publish", "manual_review", "reject"],
}

PROMPT_TEMPLATE = """你是一个严格的知识库发布审核员。

下面有 {count} 条候选知识条目
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

{item_block}

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


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path, records):
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def pre_flags(row):
    flags = []
    text = row["question"] + "\n" + row["answer"]
    if PHONE_PATTERN.search(text):
        flags.append("PRIVACY_PHONE")
    if len(row["answer"]) < 20:
        flags.append("ANSWER_TOO_SHORT")
    return flags


def clamp(judgment):
    for field, values in ENUMS.items():
        if judgment.get(field) not in values:
            judgment[field] = "manual_review" if field == "verdict" else values[-1]
    judgment["item_index"] = int(judgment.get("item_index"))
    if not isinstance(judgment.get("notes"), list):
        judgment["notes"] = []
    return judgment


def call_llm(api, prompt):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = api.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as error:  # noqa: BLE001
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM 调用失败: {last_error}")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")
    api = OpenAI(base_url=BASE_URL, api_key=os.environ["OPENAI_API_KEY"], timeout=180)

    source = {row["issue_key"]: row for row in load_jsonl(DEDUP_FILE)}
    candidates = [
        {**source[record["issue_key"]], "llm_filter": record}
        for record in load_jsonl(FILTER_FILE)
        if record["kb_worthy"] == "candidate"
    ]
    for item in candidates:
        item["pre_flags"] = pre_flags(item)
    indexed = list(enumerate(candidates))
    results_done = {record["issue_key"] for record in load_jsonl(GATE_FILE)}
    starts = [start for start in range(0, len(indexed), BATCH_SIZE)
              if any(indexed[i][1]["issue_key"] not in results_done
                     for i in range(start, min(start + BATCH_SIZE, len(indexed))))]
    print(f"gate 待判定批次 {len(starts)} (batch={BATCH_SIZE}, workers=6)")

    def process(start):
        batch = [(index, item) for index, item in indexed[start:start + BATCH_SIZE]
                 if item["issue_key"] not in results_done]
        lines = []
        for index, item in batch:
            pre = ("存在前置标记: " + ", ".join(item["pre_flags"])) if item["pre_flags"] else "无"
            lines.append(
                f"### item_index={index}\n"
                f"issue_date: {item['issue_date']}\n"
                f"resolution: {item['resolution']}\n"
                f"knowledge_value: {item['knowledge_value']}\n"
                f"problem_type: {item['problem_type']}\n"
                f"{pre}\n"
                f"问题: {item['question']}\n"
                f"答案: {item['answer'][:ANSWER_PREVIEW_CHARS]}"
            )
        prompt = PROMPT_TEMPLATE.format(count=len(batch), item_block="\n\n".join(lines))
        judgments = None
        last_error = None
        for attempt in range(MAX_RETRIES):
            raw = call_llm(api, prompt)
            match = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if match:
                try:
                    judgments = json.loads(match.group())["judgments"]
                    break
                except (json.JSONDecodeError, KeyError) as error:
                    last_error = error
            else:
                last_error = RuntimeError("无 JSON 输出")
            time.sleep(2 ** attempt)
        if judgments is None:
            raise RuntimeError(f"batch 起 {start} JSON 解析失败（重试 {MAX_RETRIES} 次）: {last_error}")
        records = []
        for judgment in judgments:
            judgment = clamp(judgment)
            item = candidates[judgment["item_index"]]
            records.append({
                "issue_key": item["issue_key"],
                **{field: judgment[field] for field in (*ENUMS, "reason", "notes")},
                "pre_flags": item["pre_flags"],
                "model": MODEL,
            })
        return start, records

    finished = len(load_jsonl(GATE_FILE))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for start, records in executor.map(process, starts):
            append_jsonl(GATE_FILE, records)
            finished += len(records)
            print(f"  累计 {finished}", flush=True)

    results = load_jsonl(GATE_FILE)
    counts = {verdict: sum(1 for r in results if r["verdict"] == verdict) for verdict in ENUMS["verdict"]}
    print(f"完成: {counts}")


if __name__ == "__main__":
    main()
