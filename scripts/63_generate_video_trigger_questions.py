#!/usr/bin/env python3
"""Step V2 - 视频触发问题生成（LLM + deterministic grounding 门）。

对 22 个培训视频逐个生成"客户触发问题"：
客户在什么情况下会需要这个视频（客服问答检索用的 question 面）。

LLM 任务（每视频一次调用，glm-5.3-flash，配置同 54）：
- 输入：标题 / 摘要 / 分类 / 时长 + 带编号的字幕全文
- 输出 JSON：5-8 条客户视角口语化问题，
  每条附支撑字幕 segment 编号与 <=30 字提示

deterministic grounding 门（逐条问题）：
- segment_refs 必须全部存在且非空
- 问题字符 bigram 覆盖率 >= 0.60
  （对照 title + summary + 被引 segment 文本，
  防止发明视频里没有的功能点）
- 问题长度 8-40 字
- 视频内规范化去重

被拒问题保留在输出中（gate=reject + 原因），
下游（64/65）只使用 gate=pass 的问题。

断点续跑：
- 每视频完成即追加写输出 jsonl
- 重跑自动跳过已有视频；--video <id> 强制重做

用法：

    # 无 LLM 预览（校验输入 / 打印首个 prompt）
    .venv/bin/python scripts/63_generate_video_trigger_questions.py --dry-run

    # 正式运行（需要 OPENAI_API_KEY）
    .venv/bin/python scripts/63_generate_video_trigger_questions.py

    # 重做单个视频
    .venv/bin/python scripts/63_generate_video_trigger_questions.py --video video-007
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from openai import OpenAI
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "output"

INVENTORY_FILE = OUTPUT_DIR / "video_inventory_v1.jsonl"
OUTPUT_JSONL = OUTPUT_DIR / "video_trigger_questions_v1.jsonl"
OUTPUT_XLSX = OUTPUT_DIR / "video_trigger_questions_v1.xlsx"
ERROR_LOG = OUTPUT_DIR / "video_trigger_questions_llm_errors.log"

VIDEO_ROOT = Path("/Users/qiyue/Desktop/video_learning")
SRT_DIR = VIDEO_ROOT / "learning_center" / "transcripts"

MODEL = "glm-5.3-flash"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 240

MIN_QUESTIONS = 3
MAX_QUESTIONS = 10
QUESTION_MIN_LEN = 8
QUESTION_MAX_LEN = 40
# gate v2: 全池（title+summary+全部字幕）覆盖率阈值。
# 口语改写会使"问题 vs 被引段落"的字面重合降到 0.1-0.5，
# 发明特征词则落在 0-0.15（全池中完全无该特征词），
# 0.40 在两者之间保留拦截带（v1 的 refs 池 0.60 已证实过严，
# 124/172 误杀，见 output/video_trigger_questions_v1.xlsx）。
GROUNDING_THRESHOLD = 0.40

# bigram 过滤的虚词字（不参与 grounding 判定）
STOP_CHARS = set("的了了吗呢吧啊嘛么呀哦喔是就把被让向从在和与或及对为")


def norm_text(s: str) -> str:
    return re.sub(r"[\s，。、；：？！“”‘’（）,.:;?!()\-—…]", "", s)


def parse_srt_segments(path: Path):
    """返回 [(seg_no, text)]，seg_no 从 1 开始。"""
    raw = path.read_text(encoding="utf-8")
    segments = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [l for l in block.split("\n") if l.strip()]
        ts, text_rows = None, []
        for line in lines:
            if "-->" in line:
                ts = line.strip()
            elif not line.strip().isdigit():
                text_rows.append(line.strip())
        if ts and text_rows:
            segments.append("".join(text_rows))
    return [(i + 1, t) for i, t in enumerate(segments)]


def load_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------- LLM schema

class TriggerQuestion(BaseModel):
    question: str = Field(..., min_length=1)
    segment_refs: List[int]
    hint: str = Field(..., min_length=1)


class TriggerQuestionSet(BaseModel):
    video_id: str
    questions: List[TriggerQuestion]


def strip_json_fences(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def log_llm_error(video_id, raw, error):
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write(
            "=" * 70 + "\n"
            + f"[{datetime.now().isoformat()}] {video_id}\n"
            + f"ERROR: {error}\n"
            + "RAW RESPONSE:\n" + raw + "\n"
        )


def build_prompt(video, segments):
    seg_lines = "\n".join(
        f"[{no}] {text}" for no, text in segments
    )
    duration_min = video["duration_sec"] / 60
    return f"""你是客服知识库工程师。下面是一个 CRM 产品的客户培训视频资料。

视频标题：{video['title']}
视频分类：{video['category']}
视频摘要：{video['summary']}
视频时长：{duration_min:.1f} 分钟

字幕全文（[n] 为段落编号）：

{seg_lines}

请站在客户/门店使用者的视角，生成 5-8 条"触发问题"：
客户在什么情况下会需要看这个视频（即客户会怎么问）。

要求：
1. 问题必须是客户真实会问的口语化问法（例如"保养到期的客户要怎么跟进？"），不要照抄视频标题；
2. 每条问题必须标注支撑它的字幕段落编号（segment_refs，1-3 个），问题内容必须来自这些段落；
3. 问题只能涉及字幕和摘要中实际讲到的功能，绝对不能发明视频没有的功能或说法；
4. 覆盖视频的主要功能点，不要所有问题都围绕同一个细节；
5. 每条问题附一句不超过 30 字的提示（hint），说明这个视频能解答什么。

只输出 JSON，格式：
{{
  "video_id": "{video['video_id']}",
  "questions": [
    {{"question": "...", "segment_refs": [1, 2], "hint": "..."}}
  ]
}}"""


def generate_for_video(client, video, segments):
    prompt = build_prompt(video, segments)
    expected_id = video["video_id"]

    last_error = None
    repair_note = None
    last_raw = None

    for attempt in range(1, MAX_RETRIES + 1):
        started = time.time()
        messages = [{"role": "user", "content": prompt}]
        if repair_note and last_raw:
            messages.append({"role": "assistant", "content": last_raw})
            messages.append({"role": "user", "content": repair_note})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            extra_body={"reasoning_effort": "low"},
        )
        raw = strip_json_fences(response.choices[0].message.content)
        last_raw = raw

        try:
            data = json.loads(raw)
            parsed = TriggerQuestionSet.model_validate(data)
            if parsed.video_id != expected_id:
                raise ValueError(
                    f"video_id 不匹配: 期望 {expected_id}, "
                    f"实际 {parsed.video_id}"
                )
            if not (MIN_QUESTIONS <= len(parsed.questions) <= MAX_QUESTIONS):
                raise ValueError(
                    f"questions 数量 {len(parsed.questions)} 不在 "
                    f"[{MIN_QUESTIONS}, {MAX_QUESTIONS}]"
                )
            if attempt > 1:
                log_llm_error(expected_id, raw,
                              f"RECOVERED_ON_ATTEMPT_{attempt}")
            return parsed, attempt, round(time.time() - started, 2)
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            log_llm_error(expected_id, raw, last_error)
            repair_note = (
                "你上一次的输出未通过校验，错误如下:\n"
                f"{last_error}\n\n请重新输出完整 JSON，要求:\n"
                f"1. video_id 必须是 {expected_id};\n"
                f"2. questions 数量在 {MIN_QUESTIONS}-{MAX_QUESTIONS} 之间;\n"
                "3. 每条含 question / segment_refs / hint 三个字段,\n"
                "   segment_refs 是字幕段落编号数组;\n"
                "4. 不要输出 JSON 以外的任何文字。"
            )

    raise RuntimeError(
        f"{expected_id}: LLM 连续 {MAX_RETRIES} 次未通过校验, "
        f"最后错误: {last_error}"
    )


# ------------------------------------------------------------- grounding 门

def bigrams(s: str):
    return {
        (s[i], s[i + 1])
        for i in range(len(s) - 1)
        if s[i] not in STOP_CHARS and s[i + 1] not in STOP_CHARS
    }


def gate_question(q, video, seg_text_by_no, seen_norm, full_pool_bigrams):
    """gate v2。返回 (gate, reason, grounding_ratio)。

    - 长度 / 视频内去重 / refs 存在性（同 v1）
    - FULL_POOL_GROUNDING: 问题 bigram 对
      title+summary+全部字幕 的覆盖率 >= GROUNDING_THRESHOLD
      （容忍口语改写，拦截发明特征词）
    - REFS_UNRELATED: 被引段落必须与问题共享
      至少 1 个内容 bigram（防乱引）
    """
    question = q.question.strip()
    if not (QUESTION_MIN_LEN <= len(question) <= QUESTION_MAX_LEN):
        return "reject", "LENGTH_OUT_OF_RANGE", 0.0

    qn = norm_text(question)
    if qn in seen_norm:
        return "reject", "DUPLICATE_IN_VIDEO", 0.0

    refs = sorted(set(q.segment_refs))
    if not refs:
        return "reject", "EMPTY_SEGMENT_REFS", 0.0
    bad_refs = [r for r in refs if r not in seg_text_by_no]
    if bad_refs:
        return "reject", f"INVALID_SEGMENT_REFS:{bad_refs}", 0.0

    q_bigrams = bigrams(qn)
    if not q_bigrams:
        return "reject", "NO_CONTENT_BIGRAMS", 0.0

    ratio = len(q_bigrams & full_pool_bigrams) / len(q_bigrams)
    if ratio < GROUNDING_THRESHOLD:
        return "reject", f"LOW_GROUNDING:{ratio:.2f}", round(ratio, 4)

    ref_text = norm_text(
        "".join(seg_text_by_no[r] for r in refs)
    )
    if not (q_bigrams & bigrams(ref_text)):
        return "reject", "REFS_UNRELATED", round(ratio, 4)

    return "pass", "", round(ratio, 4)


def gate_video_questions(question_dicts, video, seg_text_by_no,
                         full_pool_bigrams):
    """对一批问题 dict（question/segment_refs/hint）跑门，返回 gated 列表。"""
    seen_norm = set()
    gated = []
    for qd in question_dicts:
        q = TriggerQuestion(
            question=qd["question"],
            segment_refs=qd["segment_refs"],
            hint=qd["hint"],
        )
        gate, reason, ratio = gate_question(
            q, video, seg_text_by_no, seen_norm, full_pool_bigrams
        )
        if gate == "pass":
            seen_norm.add(norm_text(q.question))
        gated.append({
            "question": q.question.strip(),
            "segment_refs": sorted(set(q.segment_refs)),
            "hint": q.hint.strip(),
            "gate": gate,
            "reason": reason,
            "grounding": ratio,
        })
    return gated


def write_xlsx(records):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    total_q = sum(len(r["questions"]) for r in records)
    pass_q = sum(
        1 for r in records for x in r["questions"] if x["gate"] == "pass"
    )
    ws.append(["videos", len(records)])
    ws.append(["questions_total", total_q])
    ws.append(["questions_pass", pass_q])
    ws.append(["questions_reject", total_q - pass_q])
    for r in records:
        ws.append([
            r["video_id"], r["title"],
            f"pass {sum(1 for x in r['questions'] if x['gate'] == 'pass')}"
            f" / {len(r['questions'])}",
            f"attempt {r.get('attempt', '')}",
        ])

    ws = wb.create_sheet("questions")
    ws.append(["video_id", "title", "gate", "reason", "grounding",
               "question", "segment_refs", "hint"])
    for r in records:
        for x in r["questions"]:
            ws.append([
                r["video_id"], r["title"], x["gate"], x["reason"],
                x["grounding"], x["question"],
                ",".join(str(i) for i in x["segment_refs"]),
                x["hint"],
            ])

    wb.save(OUTPUT_XLSX)


def run_recheck():
    """不调用 LLM：对已有输出按当前门规则重算判定并覆盖写回。

    只重算 gate/reason/grounding 三个判定字段，
    question / segment_refs / hint 逐字不变。
    """
    records = load_jsonl(OUTPUT_JSONL)
    if not records:
        raise RuntimeError(f"{OUTPUT_JSONL} 为空，先运行生成模式")
    if len(records) != 22:
        raise RuntimeError(
            f"应 22 个视频, 实际 {len(records)}"
        )

    total, passed = 0, 0
    for rec in records:
        vid = rec["video_id"]
        srt_path = SRT_DIR / f"{vid}.srt"
        if not srt_path.exists():
            raise RuntimeError(f"{vid}: 缺少 SRT {srt_path}")
        segments = parse_srt_segments(srt_path)
        seg_by_no = {no: text for no, text in segments}
        full_pool = bigrams(norm_text(
            rec["title"] + rec["summary"]
            + "".join(text for _, text in segments)
        ))
        old = [(q["gate"], q["reason"], q["grounding"])
               for q in rec["questions"]]
        rec["questions"] = gate_video_questions(
            rec["questions"], rec, seg_by_no, full_pool
        )
        new = [(q["gate"], q["reason"], q["grounding"])
               for q in rec["questions"]]
        if old != new:
            changed = sum(1 for a, b in zip(old, new) if a != b)
            print(f"{vid}: {changed} 条判定变化")
        total += len(rec["questions"])
        passed += sum(1 for q in rec["questions"]
                      if q["gate"] == "pass")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in sorted(records, key=lambda r: r["video_id"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_xlsx(records)
    print(f"recheck done: pass {passed}/{total}")
    print(f"wrote: {OUTPUT_JSONL}")
    print(f"wrote: {OUTPUT_XLSX}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="不调用 LLM：校验输入并打印首个 prompt")
    ap.add_argument("--video", default=None,
                    help="强制重做指定 video_id")
    ap.add_argument("--recheck", action="store_true",
                    help="不调用 LLM：对已有输出按当前门规则重算判定")
    args = ap.parse_args()

    if args.recheck:
        return run_recheck()

    if not INVENTORY_FILE.exists():
        raise RuntimeError(f"缺少输入 {INVENTORY_FILE}，先运行 62")

    inventory = {
        r["video_id"]: r
        for r in load_jsonl(INVENTORY_FILE)
    }
    if len(inventory) != 22:
        raise RuntimeError(
            f"inventory 应有 22 个视频, 实际 {len(inventory)}"
        )

    # 组装每视频的字幕段
    video_segments = {}
    for vid, video in inventory.items():
        srt_path = SRT_DIR / f"{vid}.srt"
        if not srt_path.exists():
            raise RuntimeError(f"{vid}: 缺少 SRT {srt_path}")
        segments = parse_srt_segments(srt_path)
        if not segments:
            raise RuntimeError(f"{vid}: SRT 解析为空")
        video_segments[vid] = segments

    if args.dry_run:
        first_id = sorted(inventory)[0]
        prompt = build_prompt(inventory[first_id], video_segments[first_id])
        print(f"[dry-run] videos={len(inventory)} "
              f"segments_total={sum(len(s) for s in video_segments.values())}")
        print(f"[dry-run] prompt[{first_id}] {len(prompt)} chars:")
        print(prompt[:1200])
        print("...")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY 环境变量")

    client = OpenAI(
        base_url=BASE_URL,
        api_key=os.environ.get("OPENAI_API_KEY"),
        timeout=REQUEST_TIMEOUT,
    )

    existing = {
        r["video_id"]: r for r in load_jsonl(OUTPUT_JSONL)
    }
    if args.video:
        existing.pop(args.video, None)

    todo = [
        vid for vid in sorted(inventory)
        if vid not in existing
    ]
    print(f"todo: {len(todo)} videos "
          f"(skipped {len(inventory) - len(todo)})")

    results = dict(existing)
    for i, vid in enumerate(todo, 1):
        video = inventory[vid]
        segments = video_segments[vid]
        seg_text_by_no = {no: text for no, text in segments}

        parsed, attempt, elapsed = generate_for_video(
            client, video, segments
        )

        seg_by_no = dict(video_segments[vid])
        full_pool = bigrams(norm_text(
            video["title"] + video["summary"]
            + "".join(text for _, text in segments)
        ))
        gated = gate_video_questions(
            [q.model_dump() for q in parsed.questions],
            video, seg_by_no, full_pool,
        )

        record = {
            "video_id": vid,
            "title": video["title"],
            "category": video["category"],
            "summary": video["summary"],
            "duration_sec": video["duration_sec"],
            "attempt": attempt,
            "request_seconds": elapsed,
            "model": MODEL,
            "generated_at": datetime.now().isoformat(),
            "questions": gated,
        }
        results[vid] = record

        # 每视频即写盘（断点续跑）
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for r_vid in sorted(results):
                f.write(json.dumps(results[r_vid], ensure_ascii=False) + "\n")

        passed = sum(1 for x in gated if x["gate"] == "pass")
        print(f"[{i}/{len(todo)}] {vid} {video['title']}: "
              f"pass {passed}/{len(gated)} "
              f"(attempt {attempt}, {elapsed}s)")

    write_xlsx(list(results.values()))
    print(f"\nwrote: {OUTPUT_JSONL}")
    print(f"wrote: {OUTPUT_XLSX}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
