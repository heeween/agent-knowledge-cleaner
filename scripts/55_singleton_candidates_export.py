#!/usr/bin/env python3
"""
Step 15.5 - Singleton funnel 汇总与 candidate 池导出
====================================================

合并四层漏斗结果, 导出 singleton candidate 池:

    3598 (singletons)
      -> A+B 硬过滤      1145  (排除 2453, 全留审计)
      -> C embedding 去重 1128  (C1 3 + C2 14)
      -> D LLM 判定       858 candidate (reject 270)

注意:

- candidate 池不是正式 KB,
  后续仍需独立 publishability gate
- 全部内容逐字来自冻结/已验证输出

守卫:

- 漏斗链路一致: candidate 必须
  同时存在于 survivors / kept / llm candidate
- candidate 答案非空
- candidate 与任何一层排除集无交集

输出:

    output/singleton_candidates_v1.jsonl / .xlsx
    output/singleton_funnel_audit_v1.xlsx
"""

import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

FUNNEL_FILE = OUTPUT_DIR / "singleton_funnel_v1.jsonl"
DEDUP_FILE = OUTPUT_DIR / "singleton_dedup_v1.jsonl"
LLM_FILE = OUTPUT_DIR / "singleton_llm_filter_v1.jsonl"

CANDIDATES_JSONL = (
    OUTPUT_DIR / "singleton_candidates_v1.jsonl"
)
CANDIDATES_XLSX = (
    OUTPUT_DIR / "singleton_candidates_v1.xlsx"
)
AUDIT_XLSX = (
    OUTPUT_DIR / "singleton_funnel_audit_v1.xlsx"
)


def load_jsonl(path: Path):

    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():

    funnel = {
        record["issue_key"]: record
        for record in load_jsonl(FUNNEL_FILE)
    }
    dedup = {
        record["issue_key"]: record
        for record in load_jsonl(DEDUP_FILE)
    }
    llm = {
        record["issue_key"]: record
        for record in load_jsonl(LLM_FILE)
    }

    # ---------- 守卫: 链路一致 ----------

    errors = []

    survivor_keys = {
        key for key, record in funnel.items()
        if not record["excluded"]
    }
    kept_keys = {
        key for key, record in dedup.items()
        if record["decision"] == "kept"
    }
    candidate_keys = {
        key for key, record in llm.items()
        if record["kb_worthy_final"] == "candidate"
    }

    if not candidate_keys <= kept_keys:
        errors.append(
            "candidate 中存在未通过 C 层的 issue"
        )

    if not candidate_keys <= survivor_keys:
        errors.append(
            "candidate 中存在未通过 A+B 的 issue"
        )

    if len(llm) != len(kept_keys):
        errors.append(
            f"LLM 判定数 {len(llm)} != C 层 kept "
            f"{len(kept_keys)}"
        )

    excluded_keys = {
        key for key, record in funnel.items()
        if record["excluded"]
    }

    if candidate_keys & excluded_keys:
        errors.append(
            "candidate 与 A+B 排除集有交集"
        )

    # ---------- 组装 candidate 池 ----------

    candidates = []

    for key in sorted(candidate_keys):
        funnel_record = funnel[key]
        llm_record = llm[key]
        dedup_record = dedup[key]

        answer = funnel_record["answer"]

        if not answer.strip():
            errors.append(
                f"{key} candidate 答案为空"
            )
            continue

        candidates.append({
            "issue_key": key,
            "question": funnel_record["question"],
            "question_normalized": funnel_record[
                "question_normalized"
            ],
            "answer": answer,
            "solution": "",
            "resolution": funnel_record[
                "resolution"
            ],
            "temporal_status": funnel_record[
                "temporal_status"
            ],
            "knowledge_value": funnel_record[
                "knowledge_value"
            ],
            "confidence": funnel_record[
                "confidence"
            ],
            "problem_type": funnel_record[
                "problem_type"
            ],
            "crm_module": funnel_record[
                "crm_module"
            ],
            "crm_feature": funnel_record[
                "crm_feature"
            ],
            "document_id": funnel_record[
                "document_id"
            ],
            "source_filename": funnel_record[
                "source_filename"
            ],
            "issue_date": funnel_record[
                "issue_date"
            ],
            "funnel_max_official_similarity": (
                dedup_record[
                    "max_official_similarity"
                ]
            ),
            "funnel_qa_aligned": llm_record[
                "qa_aligned"
            ],
            "funnel_reusable": llm_record[
                "reusable"
            ],
            "funnel_incident_risk": llm_record[
                "incident_risk"
            ],
            "funnel_stable_limitation": llm_record[
                "stable_limitation"
            ],
            "funnel_llm_reason": llm_record[
                "reason"
            ],
            "status": "candidate",
        })

    if errors:
        for error in errors:
            print(f"守卫失败: {error}")

        raise SystemExit(1)

    # ---------- 导出 ----------

    with open(
        CANDIDATES_JSONL, "w", encoding="utf-8"
    ) as output:
        for record in candidates:
            output.write(
                json.dumps(
                    record, ensure_ascii=False
                )
                + "\n"
            )

    df = pd.DataFrame(candidates)

    audit_summary = [
        {"stage": "singletons", "count": 3598},
        {
            "stage": "A+B_excluded",
            "count": sum(
                1 for r in funnel.values()
                if r["excluded"]
            ),
        },
        {"stage": "AB_survivors", "count": len(
            survivor_keys
        )},
        {
            "stage": "C1_excluded_official_dup",
            "count": sum(
                1 for r in dedup.values()
                if r["reason"]
                == "C1_DUPLICATE_OF_OFFICIAL_KB"
            ),
        },
        {
            "stage": "C2_excluded_intra_dup",
            "count": sum(
                1 for r in dedup.values()
                if r["reason"]
                == "C2_INTRA_SURVIVOR_NEAR_DUPLICATE"
            ),
        },
        {"stage": "C_kept", "count": len(kept_keys)},
        {
            "stage": "D_reject",
            "count": sum(
                1 for r in llm.values()
                if r["kb_worthy_final"] == "reject"
            ),
        },
        {
            "stage": "D_candidates",
            "count": len(candidates),
        },
        {
            "stage": "stale_rescue_backlog",
            "count": sum(
                1 for r in funnel.values()
                if r["stale"]
            ),
        },
    ]

    flat = df.drop(
        columns=["question_normalized"]
    )

    with pd.ExcelWriter(
        CANDIDATES_XLSX, engine="openpyxl"
    ) as writer:
        flat.to_excel(
            writer,
            sheet_name="candidates",
            index=False,
        )

    with pd.ExcelWriter(
        AUDIT_XLSX, engine="openpyxl"
    ) as writer:
        pd.DataFrame(audit_summary).to_excel(
            writer, sheet_name="funnel", index=False
        )
        pd.DataFrame(
            [record for record in funnel.values()
             if record["excluded"]]
        ).to_excel(
            writer,
            sheet_name="excluded_ab",
            index=False,
        )
        pd.DataFrame(
            [record for record in llm.values()
             if record["kb_worthy_final"]
             == "reject"]
        ).to_excel(
            writer,
            sheet_name="rejected_d",
            index=False,
        )

    print("=" * 60)
    print("Singleton funnel 汇总完成（Step 15.5）")
    print("=" * 60)

    for row in audit_summary:
        print(f"  {row['stage']:<28} "
              f"{row['count']}")

    print()
    print(f"candidate 池: {len(candidates)}")
    print(f"输出: {CANDIDATES_JSONL}")
    print(f"输出: {CANDIDATES_XLSX}")
    print(f"输出: {AUDIT_XLSX}")


if __name__ == "__main__":
    main()
