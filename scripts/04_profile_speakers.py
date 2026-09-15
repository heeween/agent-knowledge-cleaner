from pathlib import Path
from collections import Counter, defaultdict
import json
import pandas as pd


MESSAGES_FILE = Path("output/messages.jsonl")

OUTPUT_FILE = Path(
    "output/speaker_profile.xlsx"
)


def main():

    speaker_count = Counter()
    speaker_documents = defaultdict(set)

    with open(
        MESSAGES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            message = json.loads(line)

            speaker = message["speaker"]
            document_id = message["document_id"]

            speaker_count[speaker] += 1
            speaker_documents[speaker].add(
                document_id
            )

    rows = []

    for speaker, count in speaker_count.most_common():

        rows.append({
            "speaker": speaker,
            "message_count": count,
            "document_count": len(
                speaker_documents[speaker]
            ),
        })

    df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("说话人统计完成")
    print("=" * 70)

    print(
        f"说话人数量：{len(df)}"
    )

    print()

    print(
        df.head(50).to_string(
            index=False
        )
    )

    print()
    print(
        f"完整结果：{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()