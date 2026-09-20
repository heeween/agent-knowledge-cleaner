#!/usr/bin/env python3
"""
Step 18.5 - funnel v2 Layer D: LLM 准入过滤
============================================

与 scripts/54 同 prompt、同批量 (10)、同判定维度,
仅输入输出文件不同 (2026-03-01 之后子集)。

    input:  output/funnel_v2_dedup_survivors.jsonl
    output: output/funnel_v2_llm_filter.jsonl  (全量审计, 断点续跑)
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
RESULT_FILE = OUTPUT_DIR / "funnel_v2_llm_filter.jsonl"

MODEL = "glm-5.3-flash"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
BATCH_SIZE = 10
ANSWER_PREVIEW_CHARS = 600
MAX_RETRIES = 4

ENUMS = {
    "qa_aligned": ["yes", "partial", "no"],
    "reusable": ["yes", "partial", "no"],
    "incident_risk": ["none", "low", "high"],
    "kb_worthy": ["candidate", "reject"],
}

PROMPT_TEMPLATE = """你是一个严格的知识库准入审核员。

下面有 {count} 条从客服聊天中抽取的问答
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
- issue_date 仅作背景参考, 不因日期旧而 reject
- 宁可漏选不可误收

## 待判定条目

{item_block}

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


def clamp(judgment):
    for field, values in ENUMS.items():
        if judgment.get(field) not in values:
            judgment[field] = "reject" if field == "kb_worthy" else values[-1]
    judgment["stable_limitation"] = bool(judgment.get("stable_limitation"))
    judgment["item_index"] = int(judgment.get("item_index"))
    return judgment


def item_block(items, start, end):
    lines = []
    for index in range(start, end):
        item = items[index]
        lines.append(
            f"### item_index={index}\n"
            f"issue_key: {item['issue_key']}\n"
            f"issue_date: {item['issue_date']}\n"
            f"resolution: {item['resolution']}\n"
            f"knowledge_value: {item['knowledge_value']}\n"
            f"问题: {item['question']}\n"
            f"答案: {item['answer'][:ANSWER_PREVIEW_CHARS]}"
        )
    return "\n\n".join(lines)


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

    items = [row for row in load_jsonl(DEDUP_FILE) if row.get("dedup_decision") == "kept"]
    if not items:
        raise SystemExit("输入为空: 先运行 scripts/72_dedup_v2.py")
    done = {record["issue_key"] for record in load_jsonl(RESULT_FILE)}
    starts = [start for start in range(0, len(items), BATCH_SIZE)
              if any(items[index]["issue_key"] not in done
                     for index in range(start, min(start + BATCH_SIZE, len(items))))]
    print(f"待判定批次 {len(starts)} (batch={BATCH_SIZE}, workers=6)")

    def process(start):
        end = min(start + BATCH_SIZE, len(items))
        prompt = PROMPT_TEMPLATE.format(count=end - start, item_block=item_block(items, start, end))
        raw = call_llm(api, prompt)
        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            raise RuntimeError(f"batch [{start},{end}) 无 JSON 输出")
        records = []
        for judgment in json.loads(match.group())["judgments"]:
            judgment = clamp(judgment)
            item = items[judgment["item_index"]]
            records.append({
                "issue_key": item["issue_key"],
                **{field: judgment[field] for field in (*ENUMS, "stable_limitation", "reason")},
                "model": MODEL,
            })
        return start, end, records

    finished = len(load_jsonl(RESULT_FILE))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        for start, end, records in executor.map(process, starts):
            append_jsonl(RESULT_FILE, records)
            finished += len(records)
            print(f"  [{end}/{len(items)}] 累计 {finished}", flush=True)

    results = load_jsonl(RESULT_FILE)
    candidates = sum(1 for r in results if r["kb_worthy"] == "candidate")
    print(f"完成: {len(results)} 行判定, candidate {candidates}")


if __name__ == "__main__":
    main()
