from pathlib import Path
import pandas as pd


INVENTORY_FILE = Path("output/document_inventory.xlsx")
OUTPUT_FILE = Path("output/duplicate_groups.xlsx")


def main():
    df = pd.read_excel(INVENTORY_FILE)

    # 按 SHA256 分组
    duplicated = df[
        df["sha256"].duplicated(keep=False)
    ].copy()

    if duplicated.empty:
        print("没有发现完全重复的文档。")
        return

    # 给每个重复组编号
    duplicated["duplicate_group"] = (
        duplicated
        .groupby("sha256")
        .ngroup()
        .apply(lambda x: f"G{x + 1:04d}")
    )

    # 调整列顺序
    columns = [
        "duplicate_group",
        "document_id",
        "filename",
        "title",
        "file_size",
        "word_count",
        "modified_time",
        "sha256",
    ]

    duplicated = duplicated[columns]

    # 按重复组排序
    duplicated = duplicated.sort_values(
        ["duplicate_group", "document_id"]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    duplicated.to_excel(
        OUTPUT_FILE,
        index=False
    )

    # 统计
    group_count = duplicated["duplicate_group"].nunique()
    duplicate_file_count = len(duplicated)
    extra_file_count = duplicate_file_count - group_count

    print()
    print("=" * 60)
    print("精确重复检测完成")
    print("=" * 60)
    print(f"重复组数量：{group_count}")
    print(f"涉及文件：{duplicate_file_count}")
    print(f"可消除的重复文件：{extra_file_count}")
    print(f"结果文件：{OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()