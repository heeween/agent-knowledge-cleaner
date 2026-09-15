#!/usr/bin/env python3
"""
Step 13.3 - 正式文档证据关联
============================

把 21 份正式 CRM 文档解析出的 90 个证据块
(FD-CH, authority = official_training)
与 5 个目标 mixed cluster 关联:

    QCLUSTER-0006  开通短信发送功能
                   (insufficient_evidence, blocked)
    QCLUSTER-0074  回访工单没有录音
                   (INCIDENT_TEMPORARY, blocked)
    QCLUSTER-0139  企业微信车主画像空白
                   (INCIDENT_TEMPORARY, blocked)
    QCLUSTER-0194  登录验证过期/验证码无效
                   (STALE, blocked)
    QCLUSTER-0008  CRM 无法登录处理
                   (CURRENT_CONFIRMED, 已进 v5)

用途 (PROJECT_HANDOFF 17 / 27):

- conflict resolution
- temporal version resolution
- current product rule validation
- final publishability gate

原则:

- 文档证据只允许逐字引用 chunk text
- 聊天证据仅用于理解 cluster 的争议点,
  不允许把聊天内容写进对文档的引用
- 宁可漏报不可误报
- 本步骤只产出证据关联清单,
  不做仲裁决定 (仲裁在后续步骤)

结构:

1. deterministic 组装
   (chunk 全集 + cluster 争议上下文, 均来自冻结输出)
2. LLM (glm-5.3-flash) 逐 cluster 判定
   + schema 修复重试
3. deterministic grounding 门:
   - chunk_id 必须存在于 90 chunk 集合
   - quote 必须是对应 chunk text 的逐字子串
   - 同一 cluster 内 chunk_id 去重 (保留首条)
   - quote 全部失败的 link 丢弃 (留 gate 事件)

输出 (不修改任何冻结输入):

    output/formal_doc_evidence_links_v1.jsonl
    output/formal_doc_evidence_links_v1.xlsx

用法:

    .venv/bin/python scripts/48_link_formal_doc_evidence.py
    .venv/bin/python scripts/48_link_formal_doc_evidence.py --dry-run
    .venv/bin/python scripts/48_link_formal_doc_evidence.py --parse-check
    .venv/bin/python scripts/48_link_formal_doc_evidence.py --only QCLUSTER-0008
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator


# ============================================================
# 配置
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

CHUNKS_FILE = (
    OUTPUT_DIR / "formal_doc_evidence_chunks_v1.jsonl"
)
AUDITS_FILE = OUTPUT_DIR / "mixed_cluster_audits_v2.jsonl"
STRUCTURE_FILE = (
    OUTPUT_DIR / "mixed_structure_classifications_v3_1.jsonl"
)
TEMPORAL_FILE = (
    OUTPUT_DIR / "temporal_validity_decisions.jsonl"
)
ISSUES_FILE = OUTPUT_DIR / "extracted_issues.jsonl"

OUTPUT_JSONL = (
    OUTPUT_DIR / "formal_doc_evidence_links_v1.jsonl"
)
OUTPUT_XLSX = (
    OUTPUT_DIR / "formal_doc_evidence_links_v1.xlsx"
)
ERROR_LOG = (
    OUTPUT_DIR / "formal_doc_evidence_links_llm_errors.log"
)

TARGET_CLUSTERS = [
    "QCLUSTER-0006",
    "QCLUSTER-0074",
    "QCLUSTER-0139",
    "QCLUSTER-0194",
    "QCLUSTER-0008",
]

MODEL = "glm-5.3-flash"
MAX_RETRIES = 4
REQUEST_TIMEOUT = 180

# 争议上下文里聊天 answer 的截断长度
CHAT_ANSWER_PREVIEW_CHARS = 300
# v3.1 structure reason 的截断长度
STRUCTURE_REASON_CHARS = 400


# ============================================================
# 输入加载
# ============================================================

def load_jsonl(path: Path):
    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def load_cluster_contexts():
    audits = {
        record["cluster_id"]: record
        for record in load_jsonl(AUDITS_FILE)
    }
    structures = {
        record["cluster_id"]: record
        for record in load_jsonl(STRUCTURE_FILE)
    }
    temporal = {
        record["cluster_id"]: record
        for record in load_jsonl(TEMPORAL_FILE)
    }
    issues = {}

    for record in load_jsonl(ISSUES_FILE):
        key = record.get("issue_id") or record.get(
            "issue_key"
        )

        if key:
            issues[key] = record

    contexts = {}

    for cluster_id in TARGET_CLUSTERS:
        audit = audits.get(cluster_id)

        if audit is None:
            raise KeyError(
                f"{cluster_id} 不在 mixed_cluster_audits_v2"
            )

        member_rows = []

        for member in audit.get("members", []):
            issue_key = (
                member.get("issue_key")
                if isinstance(member, dict)
                else member
            )
            issue = issues.get(issue_key, {})
            answer = issue.get("answer") or ""

            member_rows.append({
                "issue_key": issue_key,
                "question": issue.get("question") or "",
                "answer_preview": answer[
                    :CHAT_ANSWER_PREVIEW_CHARS
                ],
            })

        structure = structures.get(cluster_id, {})
        temporal_record = temporal.get(cluster_id)

        claim = ""

        if temporal_record:
            claim = (
                temporal_record.get("decision", {})
                .get("knowledge_claim")
                or ""
            )

        contexts[cluster_id] = {
            "cluster_id": cluster_id,
            "core_question": audit.get("core_question")
            or structure.get("canonical_question")
            or "",
            "knowledge_structure": structure.get(
                "knowledge_structure"
            ),
            "temporal_dependency": structure.get(
                "temporal_dependency"
            ),
            "structure_reason": (
                structure.get("reason") or ""
            )[:STRUCTURE_REASON_CHARS],
            "temporal_final_decision": (
                temporal_record.get("final_decision")
                if temporal_record
                else None
            ),
            "knowledge_claim": claim,
            "members": member_rows,
        }

    return contexts


# ============================================================
# Prompt 构建
# ============================================================

def build_chunks_block(chunks: list) -> str:
    lines = []

    for chunk in chunks:
        lines.append(
            f"[{chunk['chunk_id']}] "
            f"{chunk['document_id']} | "
            f"{chunk['title']}"
        )
        lines.append(chunk["text"])
        lines.append("")

    return "\n".join(lines)


def build_prompt(context: dict, chunks: list) -> str:
    member_lines = []

    for member in context["members"]:
        member_lines.append(
            f"- {member['issue_key']}\n"
            f"  聊天问题: {member['question']}\n"
            f"  聊天答案(截断): {member['answer_preview']}"
        )

    members_block = "\n".join(member_lines) or "（无）"

    temporal_line = (
        context["temporal_final_decision"]
        or "（无 temporal 决策记录）"
    )

    claim_line = context["knowledge_claim"] or "（无）"

    return f"""你是一个严格的知识库证据关联助手。

任务：从下面给出的正式产品文档证据块（chunk）中，找出与给定 cluster 争议点相关的证据块。

## cluster 争议上下文（全部来自已冻结的 pipeline 输出）

- cluster_id: {context['cluster_id']}
- 核心问题: {context['core_question']}
- Step 11.3 知识结构判定: {context['knowledge_structure']}（temporal_dependency = {context['temporal_dependency']}）
- Step 11.3 判定理由: {context['structure_reason']}
- Step 12.2 temporal 决策: {temporal_line}
- Step 12.2 knowledge_claim: {claim_line}

成员聊天证据（仅用于理解争议点，绝对不允许被引用为文档证据）:

{members_block}

## 正式文档证据块全集（authority = official_training，引用只能来自这里）

{build_chunks_block(chunks)}

## 关联规则（必须严格遵守）

1. 只能引用上面列出的 chunk_id，不允许发明编号。
2. quotes 中每条引用必须逐字摘自对应 chunk 的 text 原文：
   不允许改写、缩写、拼接、翻译，也不允许把聊天内容写进 quotes。
3. 宁可漏报不可误报：
   - direct: 该 chunk 直接回答/触及本 cluster 的核心争议点
   - supporting: 该 chunk 描述支撑争议判断的产品机制或规则
   - contextual: 仅背景相关，仅在确实有助于理解争议点时才收录
4. 不发明产品规则、菜单、条件、时间；只报告文档原文说了什么。
5. 每条 link 的 note 说明该证据对争议点的哪个具体方面有意义。
6. 若整个文档语料对该 cluster 无实质覆盖，
   links 返回空数组，并在 no_link_reason 中说明。

## 输出 JSON 格式（不要输出 JSON 以外的任何文字）

{{
  "cluster_id": "{context['cluster_id']}",
  "links": [
    {{
      "chunk_id": "FD-CH-XXXXX",
      "relevance": "direct|supporting|contextual",
      "aspect": "product_rule|operating_procedure|mechanism|temporal_hint|background",
      "quotes": ["逐字原文片段", "..."],
      "note": "该证据对争议点的意义"
    }}
  ],
  "coverage_note": "文档语料对该 cluster 争议点的覆盖情况总结",
  "no_link_reason": "links 为空时必填，否则为 null"
}}"""


# ============================================================
# LLM schema
# ============================================================

class EvidenceLink(BaseModel):

    chunk_id: str
    relevance: Literal[
        "direct", "supporting", "contextual"
    ]
    aspect: Literal[
        "product_rule",
        "operating_procedure",
        "mechanism",
        "temporal_hint",
        "background",
    ]
    quotes: List[str] = Field(min_length=1)
    note: str

    @field_validator("quotes")
    @classmethod
    def quotes_nonempty(cls, v):
        cleaned = [q for q in v if q and q.strip()]

        if not cleaned:
            raise ValueError(
                "quotes 不能全部为空"
            )

        return cleaned


class ClusterLinkResult(BaseModel):

    cluster_id: str
    links: List[EvidenceLink]
    coverage_note: str
    no_link_reason: Optional[str] = None


def normalize_llm_payload(data: dict) -> dict:

    if isinstance(data.get("links"), list):
        cleaned_links = []

        for link in data["links"]:
            if not isinstance(link, dict):
                continue

            quotes = link.get("quotes")

            if isinstance(quotes, str):
                quotes = [quotes]

            if not isinstance(quotes, list):
                quotes = []

            link["quotes"] = quotes
            cleaned_links.append(link)

        data["links"] = cleaned_links

    if data.get("no_link_reason") == "":
        data["no_link_reason"] = None

    return data


# ============================================================
# LLM 调用（schema 修复重试，模式同 43）
# ============================================================

def build_client():

    return OpenAI(
        base_url=(
            "https://open.bigmodel.cn/api/paas/v4"
        ),
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


def log_llm_error(cluster_id, raw, error):

    with ERROR_LOG.open("a", encoding="utf-8") as f:

        f.write(
            "=" * 70
            + "\n"
            + f"[{datetime.now().isoformat()}] "
            + f"{cluster_id}\n"
            + f"ERROR: {error}\n"
            + "RAW RESPONSE:\n"
            + raw
            + "\n"
        )


def build_repair_note(error):

    return (
        "你上一次的输出未通过 schema 校验，错误如下:\n"
        f"{error}\n\n"
        "请重新输出完整 JSON，要求:\n"
        "1. 字段名与类型与要求完全一致;\n"
        "2. links 内每条必须有 chunk_id / relevance /"
        " aspect / quotes / note，quotes 至少 1 条"
        "且逐字摘自对应 chunk 原文;\n"
        "3. relevance 只能取 direct/supporting/contextual,"
        " aspect 只能取 product_rule/operating_procedure/"
        "mechanism/temporal_hint/background;\n"
        "4. 不要输出 JSON 以外的任何文字。"
    )


def judge_cluster(client, context: dict, chunks: list):

    prompt = build_prompt(context, chunks)

    cluster_id = context["cluster_id"]

    last_error = None
    repair_note = None
    last_raw = None

    for attempt in range(1, MAX_RETRIES + 1):

        started = time.time()

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
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
            data = normalize_llm_payload(data)

            # cluster_id 以请求为准，
            # 防止模型回填错误编号
            data["cluster_id"] = cluster_id

            result = (
                ClusterLinkResult.model_validate(data)
            )

            if attempt > 1:

                log_llm_error(
                    cluster_id,
                    raw,
                    f"RECOVERED_ON_ATTEMPT_{attempt}",
                )

            return {
                "result": result,
                "model": MODEL,
                "attempt": attempt,
                "request_seconds": round(elapsed, 2),
                "raw": raw,
            }

        except Exception as error:

            last_error = error
            log_llm_error(cluster_id, raw, error)

            repair_note = build_repair_note(error)

            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"{cluster_id} LLM schema 校验连续 "
        f"{MAX_RETRIES} 次失败: {last_error}"
    )


# ============================================================
# deterministic grounding 门
# ============================================================

def apply_grounding_gate(context: dict, result, chunks):

    chunks_by_id = {
        chunk["chunk_id"]: chunk for chunk in chunks
    }

    kept_links = []
    gate_events = []
    dropped_quotes = []

    seen = set()

    for link in result.links:

        link_dict = link.model_dump()

        chunk_id = link_dict["chunk_id"]

        if chunk_id not in chunks_by_id:
            gate_events.append({
                "cluster_id": context["cluster_id"],
                "chunk_id": chunk_id,
                "event": "LINK_DROPPED_CHUNK_ID_UNKNOWN",
                "detail": "chunk_id 不在证据块集合中",
            })
            continue

        if chunk_id in seen:
            gate_events.append({
                "cluster_id": context["cluster_id"],
                "chunk_id": chunk_id,
                "event": "LINK_DROPPED_DUPLICATE",
                "detail": "同一 cluster 内重复 chunk_id，保留首条",
            })
            continue

        seen.add(chunk_id)

        chunk_text = chunks_by_id[chunk_id]["text"]

        kept_quotes = []

        for quote in link_dict["quotes"]:
            stripped = quote.strip()

            if stripped and stripped in chunk_text:
                kept_quotes.append(stripped)
            else:
                dropped_quotes.append({
                    "cluster_id": context["cluster_id"],
                    "chunk_id": chunk_id,
                    "quote_preview": stripped[:80],
                    "reason": (
                        "QUOTE_NOT_VERBATIM_IN_CHUNK"
                    ),
                })

        if kept_quotes:
            link_dict["quotes"] = kept_quotes
            link_dict["quote_count_dropped"] = len(
                link.quotes
            ) - len(kept_quotes)
            kept_links.append(link_dict)
        else:
            gate_events.append({
                "cluster_id": context["cluster_id"],
                "chunk_id": chunk_id,
                "event": "LINK_DROPPED_ALL_QUOTES_UNGROUNDED",
                "detail": "全部 quote 未通过逐字校验",
            })

    if not kept_links and not result.no_link_reason:
        gate_events.append({
            "cluster_id": context["cluster_id"],
            "chunk_id": "-",
            "event": "EMPTY_LINKS_WITHOUT_REASON",
            "detail": (
                "links 为空但未提供 no_link_reason"
            ),
        })

    return {
        "links": kept_links,
        "gate_events": gate_events,
        "dropped_quotes": dropped_quotes,
    }


# ============================================================
# 主流程
# ============================================================

def main():

    dry_run = "--dry-run" in sys.argv
    parse_check = "--parse-check" in sys.argv

    only = None

    if "--only" in sys.argv:
        only = sys.argv[
            sys.argv.index("--only") + 1
        ]

    target_clusters = TARGET_CLUSTERS

    if only:
        if only not in TARGET_CLUSTERS:
            raise ValueError(
                f"--only 仅支持: {TARGET_CLUSTERS}"
            )

        target_clusters = [only]

    chunks = sorted(
        load_jsonl(CHUNKS_FILE),
        key=lambda c: c["chunk_id"],
    )
    contexts = load_cluster_contexts()

    print(f"证据块: {len(chunks)}")
    print(f"目标 cluster: {target_clusters}")

    if parse_check:
        # 不调 LLM：
        # 校验 prompt 构建 + schema + gate 全链路
        fake = ClusterLinkResult(
            cluster_id="QCLUSTER-0008",
            links=[
                EvidenceLink(
                    chunk_id="FD-CH-99999",
                    relevance="direct",
                    aspect="mechanism",
                    quotes=["不存在"],
                    note="x",
                ),
                EvidenceLink(
                    chunk_id="FD-CH-00062",
                    relevance="supporting",
                    aspect="mechanism",
                    quotes=[
                        "在这里输入给员工输入一个初始密码",
                        "伪造的引用",
                    ],
                    note="x",
                ),
            ],
            coverage_note="parse-check",
        )

        gated = apply_grounding_gate(
            contexts["QCLUSTER-0008"], fake, chunks
        )

        assert len(gated["links"]) == 1
        assert len(gated["gate_events"]) == 1
        assert (
            gated["gate_events"][0]["event"]
            == "LINK_DROPPED_CHUNK_ID_UNKNOWN"
        )
        assert len(gated["dropped_quotes"]) == 1

        for cluster_id in target_clusters:
            prompt = build_prompt(
                contexts[cluster_id], chunks
            )
            print(
                f"  prompt {cluster_id}: "
                f"{len(prompt)} chars"
            )

        print("parse-check 通过")
        return

    if dry_run:
        for cluster_id in target_clusters:
            prompt = build_prompt(
                contexts[cluster_id], chunks
            )
            context = contexts[cluster_id]

            print(
                f"{cluster_id}  "
                f"members={len(context['members'])}  "
                f"structure="
                f"{context['knowledge_structure']}  "
                f"temporal="
                f"{context['temporal_final_decision']}  "
                f"prompt={len(prompt)} chars"
            )

        print(
            "dry-run 完成（未调用 LLM，未写输出）"
        )
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "缺少 OPENAI_API_KEY 环境变量"
        )

    client = build_client()

    output_records = []

    for index, cluster_id in enumerate(
        target_clusters, start=1
    ):
        context = contexts[cluster_id]

        print(
            f"[{index}/{len(target_clusters)}] "
            f"{cluster_id} 判定中..."
        )

        judged = judge_cluster(client, context, chunks)
        gated = apply_grounding_gate(
            context, judged["result"], chunks
        )

        record = {
            "cluster_id": cluster_id,
            "core_question": context["core_question"],
            "knowledge_structure": context[
                "knowledge_structure"
            ],
            "temporal_final_decision": context[
                "temporal_final_decision"
            ],
            "model": judged["model"],
            "attempt": judged["attempt"],
            "request_seconds": judged[
                "request_seconds"
            ],
            "coverage_note": judged[
                "result"
            ].coverage_note,
            "no_link_reason": judged[
                "result"
            ].no_link_reason,
            "raw_link_count": len(
                judged["result"].links
            ),
            "kept_link_count": len(gated["links"]),
            "links": gated["links"],
            "gate_events": gated["gate_events"],
            "dropped_quotes": gated["dropped_quotes"],
        }

        output_records.append(record)

        print(
            f"  raw links={record['raw_link_count']}  "
            f"kept={record['kept_link_count']}  "
            f"gate_events="
            f"{len(record['gate_events'])}  "
            f"dropped_quotes="
            f"{len(record['dropped_quotes'])}"
        )

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

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

    link_rows = []
    note_rows = []
    event_rows = []
    quote_rows = []

    for record in output_records:
        note_rows.append({
            "cluster_id": record["cluster_id"],
            "core_question": record["core_question"],
            "knowledge_structure": record[
                "knowledge_structure"
            ],
            "temporal_final_decision": record[
                "temporal_final_decision"
            ],
            "raw_link_count": record[
                "raw_link_count"
            ],
            "kept_link_count": record[
                "kept_link_count"
            ],
            "coverage_note": record["coverage_note"],
            "no_link_reason": record[
                "no_link_reason"
            ],
        })

        for link in record["links"]:
            for quote in link["quotes"]:
                link_rows.append({
                    "cluster_id": record[
                        "cluster_id"
                    ],
                    "chunk_id": link["chunk_id"],
                    "relevance": link["relevance"],
                    "aspect": link["aspect"],
                    "note": link["note"],
                    "quote": quote,
                })

        event_rows.extend(record["gate_events"])
        quote_rows.extend(record["dropped_quotes"])

    summary_rows = [
        {
            "cluster_id": record["cluster_id"],
            "raw_link_count": record[
                "raw_link_count"
            ],
            "kept_link_count": record[
                "kept_link_count"
            ],
            "gate_event_count": len(
                record["gate_events"]
            ),
            "dropped_quote_count": len(
                record["dropped_quotes"]
            ),
            "attempt": record["attempt"],
        }
        for record in output_records
    ]

    with pd.ExcelWriter(
        OUTPUT_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame(summary_rows).to_excel(
            writer, sheet_name="summary", index=False
        )
        pd.DataFrame(link_rows).to_excel(
            writer, sheet_name="links", index=False
        )
        pd.DataFrame(note_rows).to_excel(
            writer, sheet_name="cluster_notes", index=False
        )
        pd.DataFrame(event_rows).to_excel(
            writer, sheet_name="gate_events", index=False
        )
        pd.DataFrame(quote_rows).to_excel(
            writer,
            sheet_name="dropped_quotes",
            index=False,
        )

    print()
    print("=" * 60)
    print("正式文档证据关联完成（Step 13.3）")
    print("=" * 60)

    for row in summary_rows:
        print(
            f"  {row['cluster_id']}  "
            f"raw={row['raw_link_count']}  "
            f"kept={row['kept_link_count']}  "
            f"gate_events="
            f"{row['gate_event_count']}  "
            f"dropped_quotes="
            f"{row['dropped_quote_count']}"
        )

    print()
    print(f"输出: {OUTPUT_JSONL}")
    print(f"输出: {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
