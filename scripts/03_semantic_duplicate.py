from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


RAW_DIR = Path("data/raw/markdowns")
INVENTORY_FILE = Path("output/document_inventory.xlsx")
OUTPUT_FILE = Path("output/semantic_duplicate_candidates.xlsx")

MODEL_NAME = "BAAI/bge-m3"

# 候选阈值
SIMILARITY_THRESHOLD = 0.80


def load_documents(df):
    documents = []

    for _, row in df.iterrows():
        file_path = RAW_DIR / row["relative_path"]

        content = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        documents.append({
            "document_id": row["document_id"],
            "filename": row["filename"],
            "title": row["title"],
            "content": content,
        })

    return documents


def main():
    df = pd.read_excel(INVENTORY_FILE)

    print(f"文档数量：{len(df)}")

    documents = load_documents(df)

    texts = [
        item["content"]
        for item in documents
    ]

    print()
    print("正在加载 Embedding 模型...")
    model = SentenceTransformer(MODEL_NAME)

    print("开始生成 Embedding...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Embedding 完成。")

    print("开始计算文档相似度...")

    similarity_matrix = cosine_similarity(
        embeddings
    )

    results = []

    n = len(documents)

    for i in range(n):
        for j in range(i + 1, n):

            similarity = similarity_matrix[i][j]

            if similarity >= SIMILARITY_THRESHOLD:

                a = documents[i]
                b = documents[j]

                results.append({
                    "document_a": a["document_id"],
                    "document_b": b["document_id"],
                    "filename_a": a["filename"],
                    "filename_b": b["filename"],
                    "title_a": a["title"],
                    "title_b": b["title"],
                    "similarity": round(float(similarity), 4),
                })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values(
            "similarity",
            ascending=False
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_excel(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 60)
    print("语义重复候选检测完成")
    print("=" * 60)
    print(f"文档数量：{n}")
    print(f"候选关系：{len(result_df)}")
    print(f"阈值：{SIMILARITY_THRESHOLD}")
    print(f"结果：{OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()