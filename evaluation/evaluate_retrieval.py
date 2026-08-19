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

    correct = 0

    print("=" * 80)
    print("RAG RETRIEVAL EVALUATION")
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

        hit = expected_page in retrieved_pages

        if hit:
            correct += 1

        print(f"\nQuestion {index}:")
        print(question)

        print(f"Expected page: {expected_page}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Hit@{TOP_K}: {hit}")

    total = len(questions)
    recall = correct / total

    print("\n" + "=" * 80)
    print(f"Correct: {correct}/{total}")
    print(f"Recall@{TOP_K}: {recall:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
