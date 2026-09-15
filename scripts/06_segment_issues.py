#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 6 - 聊天问题 / 事件候选切分

输入：
    output/messages.jsonl
    output/speaker_roles.xlsx

输出：
    output/issue_candidates.jsonl
    output/issue_candidates.xlsx

注意：
    本步骤只做候选 issue/event 切分。
    不负责最终的问题、答案、CRM 模块抽取。
    LLM 抽取放到 Step 7。
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# 路径
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MESSAGES_FILE = ROOT_DIR / "output" / "messages.jsonl"
SPEAKER_ROLES_FILE = ROOT_DIR / "output" / "speaker_roles.xlsx"

OUTPUT_JSONL = ROOT_DIR / "output" / "issue_candidates.jsonl"
OUTPUT_XLSX = ROOT_DIR / "output" / "issue_candidates.xlsx"


# ============================================================
# 切分参数
# ============================================================

# 超过这个时间没有消息，默认认为进入新的上下文
HARD_TIME_GAP_MINUTES = 120

# 如果已经出现客服回复，
# 后面客户隔了一段时间重新提问，更可能是新 issue
NEW_QUESTION_GAP_MINUTES = 20

# 为避免长期群聊异常粘成一个巨大 issue
MAX_MESSAGES_PER_ISSUE = 30

# 太短且没有明显问题特征的候选块不输出
MIN_MESSAGES_PER_ISSUE = 2


# ============================================================
# 问句特征
# ============================================================

QUESTION_PATTERNS = [
    r"[?？]",
    r"为什么",
    r"怎么",
    r"怎样",
    r"如何",
    r"能不能",
    r"可不可以",
    r"可以不",
    r"有没有",
    r"是不是",
    r"什么原因",
    r"什么情况",
    r"哪里",
    r"在哪",
    r"多少",
    r"什么时候",
    r"咋",
    r"我这里",
    r"我这边",
    r"看不到",
    r"没显示",
    r"没有显示",
    r"打不开",
    r"不能",
    r"不行",
    r"报错",
    r"失败",
    r"怎么设置",
]

QUESTION_REGEX = [
    re.compile(pattern)
    for pattern in QUESTION_PATTERNS
]


# ============================================================
# 简单无知识价值消息
# ============================================================

NOISE_MESSAGES = {
    "",
    "好的",
    "好",
    "收到",
    "谢谢",
    "感谢",
    "嗯",
    "哦",
    "ok",
    "OK",
    "明白",
    "明白了",
    "知道了",
    "可以",
    "行",
    "是",
    "嗯嗯",
    "👌",
    "👍",
}


# ============================================================
# 工具函数
# ============================================================

def normalize_text(value):
    """
    统一处理字符串和 pandas NaN。

    Excel 空单元格读取后通常会变成 NaN，
    不能让它变成字符串 "nan"。
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def parse_timestamp(value):
    """
    尽量兼容原 messages.jsonl 中的时间格式。
    """

    value = normalize_text(value)

    if not value:
        return None

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass

    return None


def minutes_between(a, b):
    if not a or not b:
        return None

    return (b - a).total_seconds() / 60


def is_question_like(text):
    text = normalize_text(text)

    if not text:
        return False

    return any(
        regex.search(text)
        for regex in QUESTION_REGEX
    )


def is_noise_message(text):
    text = normalize_text(text)

    if text in NOISE_MESSAGES:
        return True

    # 单独 @ 某人
    if re.fullmatch(r"@\S+", text):
        return True

    # 只有表情/标点/数字等极短内容
    if len(text) <= 1:
        return True

    return False


# ============================================================
# 加载 Speaker Role
# ============================================================

def load_speaker_roles():

    if not SPEAKER_ROLES_FILE.exists():
        raise FileNotFoundError(
            f"找不到：{SPEAKER_ROLES_FILE}\n"
            "请先完成 Step 5。"
        )

    df = pd.read_excel(
        SPEAKER_ROLES_FILE,
        sheet_name="speaker_roles"
    )

    roles = {}

    for _, row in df.iterrows():

        speaker = normalize_text(row.get("speaker"))

        if not speaker:
            continue

        reviewed_role = normalize_text(
            row.get("reviewed_role")
        )

        suggested_role = normalize_text(
            row.get("suggested_role")
        )

        valid_roles = {
            "support",
            "customer",
            "internal",
            "unknown",
        }

        # 优先使用人工审核角色。
        # 只有 reviewed_role 是合法角色时才采用。
        if reviewed_role in valid_roles:
            role = reviewed_role

        elif suggested_role in valid_roles:
            role = suggested_role

        else:
            role = "unknown"

        roles[speaker] = role

    return roles


# ============================================================
# 加载消息
# ============================================================

def load_messages(speaker_roles):

    if not MESSAGES_FILE.exists():
        raise FileNotFoundError(
            f"找不到：{MESSAGES_FILE}"
        )

    messages = []

    with MESSAGES_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for source_index, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[WARN] JSON 解析失败："
                    f"messages.jsonl 第 {source_index} 行"
                )
                continue

            speaker = normalize_text(
                item.get("speaker")
            )

            content = normalize_text(
                item.get("content")
            )

            timestamp_raw = normalize_text(
                item.get("timestamp")
            )

            role = speaker_roles.get(
                speaker,
                "unknown"
            )

            messages.append({
                "source_message_index": source_index,

                "document_id": normalize_text(
                    item.get("document_id")
                ),

                "filename": normalize_text(
                    item.get("filename")
                ),

                "line_number": item.get(
                    "line_number"
                ),

                "timestamp": timestamp_raw,

                "_timestamp_dt": parse_timestamp(
                    timestamp_raw
                ),

                "speaker": speaker,
                "speaker_role": role,
                "content": content,

                "question_like": is_question_like(
                    content
                ),
            })

    return messages


# ============================================================
# 文档内排序
# ============================================================

def group_messages_by_document(messages):

    docs = defaultdict(list)

    for msg in messages:

        key = (
            msg["document_id"]
            or msg["filename"]
        )

        docs[key].append(msg)

    # 原 messages.jsonl 理论上已经有顺序，
    # 这里仍按 line_number 尽量稳定排序
    for key in docs:

        docs[key].sort(
            key=lambda x: (
                x["line_number"]
                if isinstance(
                    x["line_number"],
                    (int, float)
                )
                else 10**12,

                x["source_message_index"]
            )
        )

    return docs


# ============================================================
# Issue 状态辅助函数
# ============================================================

def issue_has_question(issue):
    return any(
        msg["question_like"]
        for msg in issue
    )


def issue_has_customer_question(issue):
    return any(
        msg["question_like"]
        and msg["speaker_role"] == "customer"
        for msg in issue
    )


def issue_has_support_reply(issue):

    return any(
        msg["speaker_role"]
        in {"support", "internal"}
        for msg in issue
    )


def issue_roles(issue):

    return {
        msg["speaker_role"]
        for msg in issue
    }


# ============================================================
# 判断是否应该切分
# ============================================================

def should_split(current_issue, msg):

    if not current_issue:
        return False, ""

    previous = current_issue[-1]

    gap = minutes_between(
        previous["_timestamp_dt"],
        msg["_timestamp_dt"]
    )

    # --------------------------------------------------------
    # Rule 1：超长时间间隔
    # --------------------------------------------------------

    if (
        gap is not None
        and gap >= HARD_TIME_GAP_MINUTES
    ):
        return (
            True,
            f"time_gap_{int(gap)}m"
        )

    # --------------------------------------------------------
    # Rule 2：块已经太大
    # --------------------------------------------------------

    if len(current_issue) >= MAX_MESSAGES_PER_ISSUE:
        return True, "max_messages"

    # --------------------------------------------------------
    # Rule 3：
    # 已经形成 “问题 -> 客服回答”
    # 后面客户重新提出明显问题，
    # 并且有一定时间间隔
    # --------------------------------------------------------

    if (
        msg["speaker_role"] == "customer"
        and msg["question_like"]
        and issue_has_customer_question(
            current_issue
        )
        and issue_has_support_reply(
            current_issue
        )
    ):

        if (
            gap is not None
            and gap >= NEW_QUESTION_GAP_MINUTES
        ):
            return (
                True,
                "new_customer_question_after_reply"
            )

    # --------------------------------------------------------
    # Rule 4：
    # 已经完成一个问答，
    # 随后另一个客户 Speaker 发起问题
    # --------------------------------------------------------

    if (
        msg["speaker_role"] == "customer"
        and msg["question_like"]
        and issue_has_support_reply(
            current_issue
        )
    ):

        customer_speakers = {
            m["speaker"]
            for m in current_issue
            if m["speaker_role"] == "customer"
        }

        if (
            customer_speakers
            and msg["speaker"]
            not in customer_speakers
        ):
            return (
                True,
                "different_customer_question"
            )

    return False, ""


# ============================================================
# 是否值得输出为 issue
# ============================================================

def should_keep_issue(issue):

    if not issue:
        return False

    meaningful = [
        msg
        for msg in issue
        if not is_noise_message(
            msg["content"]
        )
    ]

    if len(meaningful) < MIN_MESSAGES_PER_ISSUE:
        return False

    # 至少有明显问题
    if issue_has_question(issue):
        return True

    # 或者存在 customer + support 对话结构
    roles = issue_roles(issue)

    if (
        "customer" in roles
        and (
            "support" in roles
            or "internal" in roles
        )
    ):
        return True

    return False


# ============================================================
# 文档切分
# ============================================================

def segment_document(messages):

    issues = []

    current = []
    current_start_reason = "document_start"

    for msg in messages:

        if not current:
            current = [msg]
            continue

        split, reason = should_split(
            current,
            msg
        )

        if split:

            if should_keep_issue(current):

                issues.append({
                    "messages": current,
                    "end_reason": reason,
                    "start_reason": (
                        current_start_reason
                    ),
                })

            current = [msg]
            current_start_reason = reason

        else:
            current.append(msg)

    if current and should_keep_issue(current):

        issues.append({
            "messages": current,
            "end_reason": "document_end",
            "start_reason": current_start_reason,
        })

    return issues


# ============================================================
# 形成最终候选记录
# ============================================================

def build_issue_records(doc_groups):

    records = []

    issue_number = 1

    for document_key, messages in doc_groups.items():

        segmented = segment_document(messages)

        for block in segmented:

            issue_messages = block["messages"]

            first = issue_messages[0]
            last = issue_messages[-1]

            issue_id = (
                f"ISSUE-CAND-{issue_number:06d}"
            )

            speakers = []

            for msg in issue_messages:

                if msg["speaker"] not in speakers:
                    speakers.append(msg["speaker"])

            roles = sorted({
                msg["speaker_role"]
                for msg in issue_messages
            })

            question_messages = [
                msg
                for msg in issue_messages
                if msg["question_like"]
            ]

            customer_question_messages = [
                msg
                for msg in issue_messages
                if (
                    msg["question_like"]
                    and msg["speaker_role"]
                    == "customer"
                )
            ]

            source_messages = []

            for msg in issue_messages:

                source_messages.append({
                    "source_message_index":
                        msg["source_message_index"],

                    "line_number":
                        msg["line_number"],

                    "timestamp":
                        msg["timestamp"],

                    "speaker":
                        msg["speaker"],

                    "speaker_role":
                        msg["speaker_role"],

                    "content":
                        msg["content"],
                })

            # Excel 预览文本
            conversation_lines = []

            for msg in issue_messages:

                conversation_lines.append(
                    f"[{msg['timestamp']}] "
                    f"{msg['speaker']} "
                    f"({msg['speaker_role']}): "
                    f"{msg['content']}"
                )

            records.append({
                "issue_id": issue_id,

                "document_id":
                    first["document_id"],

                "source_filename":
                    first["filename"],

                "start_time":
                    first["timestamp"],

                "end_time":
                    last["timestamp"],

                "message_count":
                    len(issue_messages),

                "speaker_count":
                    len(speakers),

                "speakers":
                    speakers,

                "roles":
                    roles,

                "question_message_count":
                    len(question_messages),

                "customer_question_count":
                    len(
                        customer_question_messages
                    ),

                "has_support_reply":
                    issue_has_support_reply(
                        issue_messages
                    ),

                "start_reason":
                    block["start_reason"],

                "end_reason":
                    block["end_reason"],

                "source_messages":
                    source_messages,

                "conversation_text":
                    "\n".join(
                        conversation_lines
                    ),
            })

            issue_number += 1

    return records


# ============================================================
# 输出 JSONL
# ============================================================

def write_jsonl(records):

    OUTPUT_JSONL.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_JSONL.open(
        "w",
        encoding="utf-8"
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# 输出 Excel
# ============================================================

def write_excel(records):

    excel_rows = []

    for record in records:

        excel_rows.append({
            "issue_id":
                record["issue_id"],

            "document_id":
                record["document_id"],

            "source_filename":
                record["source_filename"],

            "start_time":
                record["start_time"],

            "end_time":
                record["end_time"],

            "message_count":
                record["message_count"],

            "speaker_count":
                record["speaker_count"],

            "speakers":
                " | ".join(
                    record["speakers"]
                ),

            "roles":
                " | ".join(
                    record["roles"]
                ),

            "question_message_count":
                record[
                    "question_message_count"
                ],

            "customer_question_count":
                record[
                    "customer_question_count"
                ],

            "has_support_reply":
                record["has_support_reply"],

            "start_reason":
                record["start_reason"],

            "end_reason":
                record["end_reason"],

            "conversation_text":
                record["conversation_text"],
        })

    df = pd.DataFrame(excel_rows)

    if len(df) == 0:
        print(
            "[WARN] 没有生成任何 issue candidate"
        )
        return

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    summary = pd.DataFrame([
        {
            "metric": "issue_count",
            "value": len(df),
        },
        {
            "metric": "avg_messages_per_issue",
            "value": round(
                df["message_count"].mean(),
                2
            ),
        },
        {
            "metric": "median_messages_per_issue",
            "value": round(
                df["message_count"].median(),
                2
            ),
        },
        {
            "metric": "max_messages_per_issue",
            "value": int(
                df["message_count"].max()
            ),
        },
        {
            "metric": "with_customer_question",
            "value": int(
                (
                    df[
                        "customer_question_count"
                    ] > 0
                ).sum()
            ),
        },
        {
            "metric": "with_support_reply",
            "value": int(
                (
                    df[
                        "has_support_reply"
                    ] == True
                ).sum()
            ),
        },
    ])

    # 前 200 个供人工快速检查
    review_df = df.head(200).copy()

    with pd.ExcelWriter(
        OUTPUT_XLSX,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="issue_candidates",
            index=False
        )

        review_df.to_excel(
            writer,
            sheet_name="review_200",
            index=False
        )

        summary.to_excel(
            writer,
            sheet_name="summary",
            index=False
        )

        workbook = writer.book

        for sheet_name in [
            "issue_candidates",
            "review_200",
            "summary",
        ]:

            ws = workbook[sheet_name]

            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

        for sheet_name in [
            "issue_candidates",
            "review_200",
        ]:

            ws = workbook[sheet_name]

            ws.column_dimensions["A"].width = 22
            ws.column_dimensions["B"].width = 18
            ws.column_dimensions["C"].width = 35
            ws.column_dimensions["D"].width = 21
            ws.column_dimensions["E"].width = 21
            ws.column_dimensions["F"].width = 14
            ws.column_dimensions["G"].width = 14
            ws.column_dimensions["H"].width = 35
            ws.column_dimensions["I"].width = 22
            ws.column_dimensions["J"].width = 22
            ws.column_dimensions["K"].width = 22
            ws.column_dimensions["L"].width = 20
            ws.column_dimensions["M"].width = 32
            ws.column_dimensions["N"].width = 32
            ws.column_dimensions["O"].width = 100

            # conversation_text 自动换行
            for row in ws.iter_rows(
                min_row=2
            ):
                row[14].alignment = (
                    row[14].alignment.copy(
                        wrap_text=True,
                        vertical="top"
                    )
                )


# ============================================================
# 统计打印
# ============================================================

def print_statistics(records):

    print("\n" + "=" * 70)
    print("Step 6 结果")
    print("=" * 70)

    print(
        f"Issue candidates："
        f"{len(records):,}"
    )

    if not records:
        return

    counts = [
        r["message_count"]
        for r in records
    ]

    customer_questions = sum(
        1
        for r in records
        if r["customer_question_count"] > 0
    )

    support_replies = sum(
        1
        for r in records
        if r["has_support_reply"]
    )

    print(
        f"平均消息数 / issue："
        f"{sum(counts) / len(counts):.2f}"
    )

    print(
        f"最大消息数："
        f"{max(counts)}"
    )

    print(
        f"包含客户问题："
        f"{customer_questions:,}"
    )

    print(
        f"包含客服/内部回复："
        f"{support_replies:,}"
    )

    print("\n消息数分布：")

    ranges = [
        (2, 3),
        (4, 6),
        (7, 10),
        (11, 20),
        (21, 30),
    ]

    for low, high in ranges:

        n = sum(
            1
            for value in counts
            if low <= value <= high
        )

        print(
            f"  {low:02d}-{high:02d}："
            f"{n:,}"
        )


# ============================================================
# main
# ============================================================

def main():

    print("=" * 70)
    print("Step 6 - 聊天问题 / 事件候选切分")
    print("=" * 70)

    print("\n读取 Speaker Roles...")

    speaker_roles = load_speaker_roles()

    print(
        f"Speaker Roles："
        f"{len(speaker_roles):,}"
    )

    role_counter = Counter(
    speaker_roles.values()
    )

    print("Speaker Role 分布：")

    for role, count in role_counter.most_common():
        print(
            f"  {role}: {count:,}"
        )

    print("\n读取消息...")

    messages = load_messages(
        speaker_roles
    )

    print(
        f"Messages："
        f"{len(messages):,}"
    )

    print("\n按文档分组...")

    doc_groups = (
        group_messages_by_document(
            messages
        )
    )

    print(
        f"Documents："
        f"{len(doc_groups):,}"
    )

    print("\n执行 issue segmentation...")

    records = build_issue_records(
        doc_groups
    )

    write_jsonl(records)
    write_excel(records)

    print_statistics(records)

    print("\n输出文件：")
    print(
        f"  {OUTPUT_JSONL}"
    )
    print(
        f"  {OUTPUT_XLSX}"
    )

    print(
        "\nStep 6 完成后，请先检查 "
        "issue_candidates.xlsx 的 review_200，"
        "暂时不要进入 Step 7。"
    )


if __name__ == "__main__":
    main()