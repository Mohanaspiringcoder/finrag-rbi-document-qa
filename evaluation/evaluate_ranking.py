import json

import chromadb
from sentence_transformers import SentenceTransformer


DB_PATH = "chroma_db"
COLLECTION_NAME = "rbi_documents"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def load_questions():
    with open("evaluation/questions.json", "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    questions = load_questions()

    recall_at_1_hits = 0
    recall_at_5_hits = 0
    reciprocal_ranks = []

    print("=" * 80)
    print("RAG RETRIEVAL RANKING EVALUATION")
    print("=" * 80)

    for index, item in enumerate(questions, start=1):

        question = item["question"]
        expected_page = item["expected_page"]

        query_embedding = model.encode([question])[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=TOP_K,
        )

        retrieved_pages = [
            metadata["page"]
            for metadata in results["metadatas"][0]
        ]

        # Recall@1
        if retrieved_pages[0] == expected_page:
            recall_at_1_hits += 1

        # Find first occurrence of expected page
        rank = None

        for position, page in enumerate(retrieved_pages, start=1):
            if page == expected_page:
                rank = position
                break

        if rank is not None:
            recall_at_5_hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

        print(f"\nQuestion {index}:")
        print(question)

        print(f"Expected page: {expected_page}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Rank of expected page: {rank}")

    total = len(questions)

    recall_at_1 = recall_at_1_hits / total
    recall_at_5 = recall_at_5_hits / total
    mrr = sum(reciprocal_ranks) / total

    print("\n" + "=" * 80)
    print(f"Recall@1 : {recall_at_1:.2%}")
    print(f"Recall@5 : {recall_at_5:.2%}")
    print(f"MRR      : {mrr:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
