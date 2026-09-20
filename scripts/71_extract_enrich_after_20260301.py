#!/usr/bin/env python3
"""
Step 18.3 - LLM 抽取/富集 2026-03-01 之后缺失切片
==================================================

两个模式 (glm-5.3-flash @ bigmodel, 与 54/56 同配置):

1. extract: 120 个从未抽取过的切片, 从 source_messages
   抽取 QA 对 (漏斗 v1 同 schema)
2. enrich: 64 行聚类路径存量 QA, 补齐 6 个漏斗元数据字段
   (置信度/知识价值/时效/解决度/模块/功能)

守卫与原则:

- answer 只允许来自 support 角色消息的逐字/合并, 禁止编造
- 枚举字段越界时保守钳制 (stable/medium/unresolved)
- 输出文件断点续跑: 已存在的 issue_key 跳过
- 不修改任何冻结输入

输出:

    output/funnel_v2_extracted.jsonl
    output/funnel_v2_enriched.jsonl
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

EXTRACT_WORKLIST = OUTPUT_DIR / "funnel_v2_extract_worklist.jsonl"
ENRICH_WORKLIST = OUTPUT_DIR / "funnel_v2_enrich_worklist.jsonl"
EXTRACTED_OUT = OUTPUT_DIR / "funnel_v2_extracted.jsonl"
ENRICHED_OUT = OUTPUT_DIR / "funnel_v2_enriched.jsonl"

MODEL = "glm-5.3-flash"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 75

TEMPORAL_ENUM = {"stable", "temporary", "future_plan", "historical"}
VALUE_ENUM = {"low", "medium", "high"}
RESOLUTION_ENUM = {"resolved", "unresolved", "partial"}

EXTRACT_PROMPT = """你是 CRM 知识库构建助手。下面是一段客服群聊切片（含说话人与角色）。
请抽取其中"客户提出的问题 + 客服给出的回答"知识对。

规则:
1. answer 只能来自 support 角色消息的原文或原文合并，禁止编造、改写事实、补充外部知识。
2. 只抽取对 CRM 产品知识有价值的对；寒暄、纯事务安排、无回答的问题跳过。
3. question_normalized 是 question 的规范化改写（去人名/指代，保留语义）。
4. temporal_status: stable=长期有效的产品逻辑; temporary=临时/一次性处理; future_plan=未来计划; historical=已过时。
5. knowledge_value: high=可复用的产品知识; medium=有一定价值; low=价值很低。
6. resolution: resolved=已解决; partial=部分解决; unresolved=未解决。
7. crm_module/crm_feature: 问题所属 CRM 模块与功能点（中文，如 车辆管理/保养到期提醒）。

只输出 JSON 数组，每个元素:
{"question": "...", "question_normalized": "...", "answer": "...",
 "confidence": 0.0-1.0, "knowledge_value": "low|medium|high",
 "temporal_status": "stable|temporary|future_plan|historical",
 "resolution": "resolved|partial|unresolved",
 "problem_type": "...", "crm_module": "...", "crm_feature": "..."}

若无可抽取内容，输出 []。"""

ENRICH_PROMPT = """你是 CRM 知识库标注助手。给定一条已抽取的问答对，
为它补齐以下元数据字段（不要改动 question/answer 本身）:

- confidence: 0.0-1.0, 回答对该问题的解决程度与可信度
- knowledge_value: "low|medium|high", 可复用产品知识价值
- temporal_status: "stable|temporary|future_plan|historical"
- resolution: "resolved|partial|unresolved"
- crm_module / crm_feature: 所属 CRM 模块与功能点（中文）

只输出 JSON 对象，键为上述 6 个字段。"""

EXTRACT_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)
ENRICH_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")
    return OpenAI(base_url=BASE_URL, api_key=os.environ["OPENAI_API_KEY"], timeout=REQUEST_TIMEOUT)


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


def call_llm(api, messages):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = api.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as error:  # noqa: BLE001 - 网络类错误统一重试
            last_error = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM 调用失败（重试 {MAX_RETRIES} 次）: {last_error}")


def clamp_meta(item):
    try:
        confidence = float(item.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.5
    item["confidence"] = min(1.0, max(0.0, confidence))
    if item.get("temporal_status") not in TEMPORAL_ENUM:
        item["temporal_status"] = "stable"
    if item.get("knowledge_value") not in VALUE_ENUM:
        item["knowledge_value"] = "medium"
    if item.get("resolution") not in RESOLUTION_ENUM:
        item["resolution"] = "unresolved"
    for field in ("question", "question_normalized", "answer", "problem_type", "crm_module", "crm_feature"):
        item[field] = str(item.get(field) or "").strip()
    return item


def run_extract(api):
    done = {row["source_candidate_id"] for row in load_jsonl(EXTRACTED_OUT)}
    slices = load_jsonl(EXTRACT_WORKLIST)
    pending = [s for s in slices if s["issue_id"] not in done]
    print(f"extract: {len(pending)}/{len(slices)} 待处理")
    buffer = []
    for index, record in enumerate(pending, 1):
        lines = [
            f"【{m['timestamp']}】{m['speaker']}({m['speaker_role']}): {m['content']}"
            for m in record["source_messages"]
        ]
        user = "群聊切片:\n" + "\n".join(lines)
        raw = call_llm(api, [{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": user}])
        match = EXTRACT_JSON_ARRAY.search(raw or "")
        items = json.loads(match.group()) if match else []
        for order, item in enumerate(items, 1):
            item = clamp_meta(item)
            if not item["question"] or not item["answer"]:
                continue
            buffer.append({
                "issue_key": f"{record['issue_id']}#NEW-{order:03d}",
                "source_candidate_id": record["issue_id"],
                "document_id": record["document_id"],
                "source_filename": record["source_filename"],
                "issue_date": record["start_time"][:10],
                **item,
                "model": MODEL,
            })
        if len(buffer) >= 20:
            append_jsonl(EXTRACTED_OUT, buffer)
            print(f"  [{index}/{len(pending)}] 累计输出 {len(buffer)} 行", flush=True)
            buffer = []
    if buffer:
        append_jsonl(EXTRACTED_OUT, buffer)
    print(f"extract 完成: 输出 {len(load_jsonl(EXTRACTED_OUT))} 行")


def run_enrich(api):
    done = {row["issue_key"] for row in load_jsonl(ENRICHED_OUT)}
    rows = load_jsonl(ENRICH_WORKLIST)
    pending = [r for r in rows if r["issue_key"] not in done]
    print(f"enrich: {len(pending)}/{len(rows)} 待处理")
    buffer = []
    for index, row in enumerate(pending, 1):
        user = json.dumps({
            "question": row["question"], "answer": row["answer"], "problem_type": row["problem_type"],
        }, ensure_ascii=False)
        raw = call_llm(api, [{"role": "system", "content": ENRICH_PROMPT}, {"role": "user", "content": user}])
        match = ENRICH_JSON_OBJECT.search(raw or "")
        meta = json.loads(match.group()) if match else {}
        enriched = dict(row)
        enriched.update(clamp_meta(meta))
        enriched["model"] = MODEL
        buffer.append(enriched)
        if len(buffer) >= 20:
            append_jsonl(ENRICHED_OUT, buffer)
            print(f"  [{index}/{len(pending)}] 累计输出 {len(buffer)} 行", flush=True)
            buffer = []
    if buffer:
        append_jsonl(ENRICHED_OUT, buffer)
    print(f"enrich 完成: 输出 {len(load_jsonl(ENRICHED_OUT))} 行")


def main():
    api = client()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "extract"):
        run_extract(api)
    if mode in ("all", "enrich"):
        run_enrich(api)


if __name__ == "__main__":
    main()
