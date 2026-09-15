#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5: Speaker 角色识别

输入:
    output/messages.jsonl

输出:
    output/speaker_roles.xlsx

目标:
    根据 Speaker 的跨文档分布、消息数量、提问特征、回答特征，
    自动给出候选角色：

    support   - 客服 / 支持
    customer  - 客户
    internal  - 产品 / 技术 / 内部人员
    unknown   - 暂时无法判断

注意:
    这里只生成候选角色，不追求 100% 自动准确。
    后续需要人工审核高频 Speaker。
"""

import json
import math
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# 路径
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MESSAGES_FILE = ROOT_DIR / "output" / "messages.jsonl"
OUTPUT_FILE = ROOT_DIR / "output" / "speaker_roles.xlsx"


# ============================================================
# 内容特征
# ============================================================

# 明显偏“客户提问”的表达
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
    r"咋",
    r"哪里",
    r"在哪",
    r"多少",
    r"什么时候",
    r"我这里",
    r"我这边",
    r"看不到",
    r"没有显示",
    r"没显示",
    r"打不开",
    r"不能",
    r"不行",
    r"报错",
    r"失败",
]


# 明显偏“客服回答 / 指导”的表达
ANSWER_PATTERNS = [
    r"这个是因为",
    r"因为",
    r"需要",
    r"建议",
    r"可以在",
    r"可以去",
    r"可以通过",
    r"你可以",
    r"您可以",
    r"你这边",
    r"您这边",
    r"在设置",
    r"设置里面",
    r"我们这边",
    r"我这边看",
    r"这边看了",
    r"看了一下",
    r"已经处理",
    r"已经修复",
    r"已经好了",
    r"处理好了",
    r"帮你",
    r"帮您",
    r"给你",
    r"给您",
    r"稍后",
    r"刷新一下",
    r"重新登录",
    r"重新进入",
    r"重新操作",
    r"操作一下",
    r"同步一下",
    r"更新一下",
    r"确认一下",
    r"检查一下",
    r"麻烦",
    r"正常的",
    r"目前是",
    r"目前这个",
    r"这个功能",
    r"系统会",
    r"系统是",
    r"逻辑是",
    r"规则是",
]


# 更偏内部产品 / 技术沟通的词
INTERNAL_PATTERNS = [
    r"接口",
    r"日志",
    r"数据库",
    r"代码",
    r"服务",
    r"服务器",
    r"发布",
    r"上线",
    r"版本",
    r"测试环境",
    r"生产环境",
    r"开发",
    r"研发",
    r"技术排查",
    r"排查",
    r"bug",
    r"BUG",
    r"需求",
    r"产品确认",
    r"产品那边",
    r"技术那边",
    r"后端",
    r"前端",
    r"SQL",
    r"sql",
]


QUESTION_REGEX = [re.compile(p) for p in QUESTION_PATTERNS]
ANSWER_REGEX = [re.compile(p) for p in ANSWER_PATTERNS]
INTERNAL_REGEX = [re.compile(p) for p in INTERNAL_PATTERNS]


# ============================================================
# 基础函数
# ============================================================

def contains_any(text, regex_list):
    """是否匹配任一规则。"""
    if not text:
        return False

    return any(regex.search(text) for regex in regex_list)


def count_matches(text, regex_list):
    """统计一条消息命中了多少个规则。"""
    if not text:
        return 0

    return sum(1 for regex in regex_list if regex.search(text))


def safe_ratio(a, b):
    if not b:
        return 0.0
    return a / b


def normalize_text(text):
    if text is None:
        return ""
    return str(text).strip()


# ============================================================
# 加载 messages.jsonl
# ============================================================

def load_messages():
    if not MESSAGES_FILE.exists():
        raise FileNotFoundError(
            f"找不到消息文件：{MESSAGES_FILE}\n"
            "请确认 Step 3 已完成，并且文件位于 output/messages.jsonl"
        )

    messages = []

    with MESSAGES_FILE.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON 解析失败，第 {line_number} 行：{e}")
                continue

            speaker = normalize_text(item.get("speaker"))
            content = normalize_text(item.get("content"))
            document_id = normalize_text(item.get("document_id"))
            filename = normalize_text(item.get("filename"))

            if not speaker:
                continue

            messages.append({
                "speaker": speaker,
                "content": content,
                "document_id": document_id,
                "filename": filename,
            })

    return messages


# ============================================================
# Speaker 聚合
# ============================================================

def build_speaker_stats(messages):
    stats = defaultdict(lambda: {
        "message_count": 0,
        "documents": set(),
        "total_length": 0,

        "question_count": 0,
        "answer_like_count": 0,
        "internal_like_count": 0,

        "question_feature_hits": 0,
        "answer_feature_hits": 0,
        "internal_feature_hits": 0,

        "sample_messages": [],
    })

    for msg in messages:
        speaker = msg["speaker"]
        content = msg["content"]

        row = stats[speaker]

        row["message_count"] += 1
        row["total_length"] += len(content)

        document_key = msg["document_id"] or msg["filename"]

        if document_key:
            row["documents"].add(document_key)

        question_hits = count_matches(content, QUESTION_REGEX)
        answer_hits = count_matches(content, ANSWER_REGEX)
        internal_hits = count_matches(content, INTERNAL_REGEX)

        row["question_feature_hits"] += question_hits
        row["answer_feature_hits"] += answer_hits
        row["internal_feature_hits"] += internal_hits

        if question_hits > 0:
            row["question_count"] += 1

        if answer_hits > 0:
            row["answer_like_count"] += 1

        if internal_hits > 0:
            row["internal_like_count"] += 1

        # 保存少量样例，方便 Excel 人工检查
        if content and len(row["sample_messages"]) < 5:
            row["sample_messages"].append(content[:150])

    return stats


# ============================================================
# 角色评分
# ============================================================

def calculate_scores(
    message_count,
    document_count,
    question_ratio,
    answer_ratio,
    internal_ratio,
):
    """
    分别计算 support / customer / internal 分数。

    这里故意采用简单、可解释的规则，
    而不是黑盒模型。

    分值不是概率。
    """

    support_score = 0.0
    customer_score = 0.0
    internal_score = 0.0

    # --------------------------------------------------------
    # 1. 跨文档数量
    #
    # 这是当前最重要的角色特征之一。
    # 出现在几十/上百个不同客户聊天中的 Speaker，
    # 非常可能是内部客服或支持人员。
    # --------------------------------------------------------

    if document_count >= 100:
        support_score += 6.0
    elif document_count >= 50:
        support_score += 5.0
    elif document_count >= 20:
        support_score += 4.0
    elif document_count >= 10:
        support_score += 2.5
    elif document_count >= 5:
        support_score += 1.0
    elif document_count <= 2:
        customer_score += 1.5

    # --------------------------------------------------------
    # 2. 消息数量
    #
    # 高频只能作为辅助条件，不能独立决定角色。
    # --------------------------------------------------------

    if message_count >= 5000:
        support_score += 3.0
    elif message_count >= 1000:
        support_score += 2.5
    elif message_count >= 500:
        support_score += 2.0
    elif message_count >= 200:
        support_score += 1.0
    elif message_count <= 10:
        customer_score += 0.3

    # --------------------------------------------------------
    # 3. 提问比例
    # --------------------------------------------------------

    if question_ratio >= 0.50:
        customer_score += 4.0
    elif question_ratio >= 0.35:
        customer_score += 3.0
    elif question_ratio >= 0.20:
        customer_score += 1.5
    elif question_ratio <= 0.08:
        support_score += 1.0

    # --------------------------------------------------------
    # 4. 回答 / 指导比例
    # --------------------------------------------------------

    if answer_ratio >= 0.40:
        support_score += 4.0
    elif answer_ratio >= 0.25:
        support_score += 3.0
    elif answer_ratio >= 0.15:
        support_score += 1.5

    # --------------------------------------------------------
    # 5. 内部技术特征
    #
    # internal 和 support 是可能重叠的。
    # internal 主要用于发现研发、产品、技术人员。
    # --------------------------------------------------------

    if internal_ratio >= 0.30:
        internal_score += 5.0
    elif internal_ratio >= 0.20:
        internal_score += 4.0
    elif internal_ratio >= 0.10:
        internal_score += 2.5
    elif internal_ratio >= 0.05:
        internal_score += 1.0

    # 跨多个群 + 技术语言，是内部人员的更强信号
    if document_count >= 10 and internal_ratio >= 0.10:
        internal_score += 2.0

    # 客户一般集中在很少的文档
    if document_count <= 3 and question_ratio >= 0.20:
        customer_score += 2.0

    return (
        round(support_score, 3),
        round(customer_score, 3),
        round(internal_score, 3),
    )


def suggest_role(
    speaker,
    message_count,
    document_count,
    question_ratio,
    answer_ratio,
    internal_ratio,
    support_score,
    customer_score,
    internal_score,
):
    """
    根据规则和分数生成候选角色。

    原则：
    1. Speaker 名称中的明确角色提示优先。
    2. 高跨文档 Speaker 更可能是 support。
    3. internal 需要比 support 更严格的证据。
    4. 小样本尽量 unknown。
    """

    speaker_lower = speaker.lower().strip()

    # ========================================================
    # 1. 特殊 / 无意义 Speaker
    # ========================================================

    generic_customer_names = {
        "客户",
        "客户方",
        "顾客",
        "用户",
        "车主",
    }

    generic_system_names = {
        "系统",
        "群助手",
        "助手",
        "机器人",
        "system",
    }

    if speaker_lower in generic_customer_names:
        return "customer", "Speaker 名称明确为客户角色"

    if speaker_lower in generic_system_names:
        return "unknown", "系统/机器人类 Speaker"

    # ========================================================
    # 2. Speaker 名称强特征
    # ========================================================

    support_name_keywords = [
        "客服",
        "值班客服",
        "售后客服",
        "运营中心",
        "客户成功",
        "客户服务",
    ]

    internal_name_keywords = [
        "研发",
        "开发",
        "技术",
        "产品经理",
        "产品部",
    ]

    if any(keyword in speaker for keyword in support_name_keywords):
        return "support", "Speaker 名称包含明确客服/运营角色"

    if any(keyword in speaker for keyword in internal_name_keywords):
        return "internal", "Speaker 名称包含明确产品/技术角色"

    # ========================================================
    # 3. 极低样本
    # ========================================================

    if message_count < 5:
        return "unknown", "样本过少"

    # ========================================================
    # 4. 超高跨文档 Speaker
    #
    # 这种情况基本不可能是普通客户。
    # ========================================================

    if document_count >= 50:
        return "support", f"跨 {document_count} 个文档，高概率内部支持人员"

    # ========================================================
    # 5. 计算排名
    # ========================================================

    scores = {
        "support": support_score,
        "customer": customer_score,
        "internal": internal_score,
    }

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_role, top_score = ranked[0]
    second_role, second_score = ranked[1]

    score_gap = top_score - second_score

    # ========================================================
    # 6. internal 要比原来严格
    #
    # 不能仅仅因为出现了“接口/bug/排查”就判断研发人员。
    # ========================================================

    if top_role == "internal":
        if message_count < 20:
            return "unknown", "internal 样本不足"

        if internal_ratio < 0.15:
            return "unknown", "内部技术特征不足"

        if internal_score < 4.0:
            return "unknown", "internal score 不足"

        # 如果 support 也很强，暂时不自动决定
        if internal_score - support_score < 1.5:
            return "unknown", "internal/support 特征接近"

        return "internal", f"internal score={internal_score:.2f}"

    # ========================================================
    # 7. 分数太低
    # ========================================================

    if top_score < 2.0:
        return "unknown", "角色特征不明显"

    # ========================================================
    # 8. 分差太小
    # ========================================================

    if score_gap < 1.0:
        return "unknown", (
            f"{top_role}/{second_role} 特征接近"
        )

    # ========================================================
    # 9. 低消息量时 support 更谨慎
    # ========================================================

    if top_role == "support":
        if message_count < 15 and document_count < 5:
            return "unknown", "support 样本不足"

    return top_role, f"{top_role} score={top_score:.2f}"
# ============================================================
# 输出
# ============================================================

def build_dataframe(stats):
    records = []

    for speaker, row in stats.items():

        message_count = row["message_count"]
        document_count = len(row["documents"])

        avg_message_length = safe_ratio(
            row["total_length"],
            message_count
        )

        question_ratio = safe_ratio(
            row["question_count"],
            message_count
        )

        answer_ratio = safe_ratio(
            row["answer_like_count"],
            message_count
        )

        internal_ratio = safe_ratio(
            row["internal_like_count"],
            message_count
        )

        (
            support_score,
            customer_score,
            internal_score,
        ) = calculate_scores(
            message_count=message_count,
            document_count=document_count,
            question_ratio=question_ratio,
            answer_ratio=answer_ratio,
            internal_ratio=internal_ratio,
        )

        suggested_role, role_reason = suggest_role(
            speaker=speaker,
            message_count=message_count,
            document_count=document_count,
            question_ratio=question_ratio,
            answer_ratio=answer_ratio,
            internal_ratio=internal_ratio,
            support_score=support_score,
            customer_score=customer_score,
            internal_score=internal_score,
        )

        # context 文件要求的 role_score：
        # 这里定义为最终最高角色分数。
        role_score = max(
            support_score,
            customer_score,
            internal_score,
        )

        records.append({
            "speaker": speaker,

            "message_count": message_count,
            "document_count": document_count,
            "avg_message_length": round(avg_message_length, 2),

            "question_count": row["question_count"],
            "answer_like_count": row["answer_like_count"],
            "internal_like_count": row["internal_like_count"],

            "question_ratio": round(question_ratio, 4),
            "answer_ratio": round(answer_ratio, 4),
            "internal_ratio": round(internal_ratio, 4),

            "question_feature_hits": row["question_feature_hits"],
            "answer_feature_hits": row["answer_feature_hits"],
            "internal_feature_hits": row["internal_feature_hits"],

            "support_score": support_score,
            "customer_score": customer_score,
            "internal_score": internal_score,

            "role_score": role_score,
            "suggested_role": suggested_role,
            "role_reason": role_reason,

            # 后面人工检查时使用
            "reviewed_role": "",
            "review_note": "",

            "sample_message_1": (
                row["sample_messages"][0]
                if len(row["sample_messages"]) > 0
                else ""
            ),
            "sample_message_2": (
                row["sample_messages"][1]
                if len(row["sample_messages"]) > 1
                else ""
            ),
            "sample_message_3": (
                row["sample_messages"][2]
                if len(row["sample_messages"]) > 2
                else ""
            ),
        })

    df = pd.DataFrame(records)

    # 默认先看消息量最大的 Speaker
    df = df.sort_values(
        by=["message_count", "document_count"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return df


def write_excel(df):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    role_summary = (
        df.groupby("suggested_role")
        .agg(
            speaker_count=("speaker", "count"),
            message_count=("message_count", "sum"),
        )
        .reset_index()
        .sort_values(
            "speaker_count",
            ascending=False
        )
    )

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            sheet_name="speaker_roles",
            index=False
        )

        role_summary.to_excel(
            writer,
            sheet_name="role_summary",
            index=False
        )

        # 单独放前 100 高频 Speaker，
        # 方便接下来人工审核
        df.head(100).to_excel(
            writer,
            sheet_name="top100_review",
            index=False
        )

        workbook = writer.book

        for sheet_name in [
            "speaker_roles",
            "role_summary",
            "top100_review",
        ]:
            ws = workbook[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

        # 给主要 sheet 设置较合理的列宽
        ws = workbook["speaker_roles"]

        widths = {
            "A": 24,   # speaker
            "B": 14,
            "C": 14,
            "D": 20,
            "E": 15,
            "F": 18,
            "G": 18,
            "H": 15,
            "I": 15,
            "J": 15,
            "K": 22,
            "L": 22,
            "M": 22,
            "N": 16,
            "O": 16,
            "P": 16,
            "Q": 14,
            "R": 18,
            "S": 28,
            "T": 18,
            "U": 28,
            "V": 50,
            "W": 50,
            "X": 50,
        }

        for col, width in widths.items():
            ws.column_dimensions[col].width = width

        ws2 = workbook["top100_review"]

        for col, width in widths.items():
            ws2.column_dimensions[col].width = width


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Step 5 - Speaker 角色识别")
    print("=" * 70)

    print(f"\n读取：{MESSAGES_FILE}")

    messages = load_messages()

    print(f"消息数量：{len(messages):,}")

    stats = build_speaker_stats(messages)

    print(f"Speaker 数量：{len(stats):,}")

    df = build_dataframe(stats)

    write_excel(df)

    print(f"\n输出：{OUTPUT_FILE}")

    print("\n角色统计：")
    print(
        df["suggested_role"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n前 20 个高频 Speaker：")

    preview_columns = [
        "speaker",
        "message_count",
        "document_count",
        "question_ratio",
        "answer_ratio",
        "internal_ratio",
        "support_score",
        "customer_score",
        "internal_score",
        "suggested_role",
    ]

    print(
        df[preview_columns]
        .head(20)
        .to_string(index=False)
    )

    print("\n完成。")
    print(
        "下一步先人工检查 output/speaker_roles.xlsx "
        "中的 top100_review，不要直接进入 Step 6。"
    )


if __name__ == "__main__":
    main()