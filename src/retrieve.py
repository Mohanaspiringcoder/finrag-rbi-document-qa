import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K,
    DISTANCE_THRESHOLD,
)


def load_vector_db():
    """Load the persistent Chroma vector database."""

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


def load_embedding_model():
    """Load the sentence-transformer embedding model."""

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def retrieve_documents(
    question,
    embedding_model,
    collection,
    top_k=TOP_K,
    distance_threshold=DISTANCE_THRESHOLD,
):
    """Retrieve relevant document chunks for a question."""

    query_embedding = embedding_model.encode(
        [question]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k,
    )

    contexts = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if distance <= distance_threshold:
            contexts.append(
                {
                    "text": document,
                    "page": metadata["page"],
                    "distance": distance,
                }
            )

    return contexts


def main():
    print("=" * 80)
    print("RBI DOCUMENT RETRIEVAL")
    print("=" * 80)

    embedding_model = load_embedding_model()
    collection = load_vector_db()

    question = input("\nQuery: ").strip()

    if not question:
        print("\nPlease enter a question.")
        return

    contexts = retrieve_documents(
        question,
        embedding_model,
        collection,
    )

    print("\nRetrieved results:")
    print("=" * 80)

    if not contexts:
        print("No relevant RBI information found.")
        return

    for i, context in enumerate(
        contexts,
        start=1,
    ):
        print("\n" + "-" * 80)
        print(f"RESULT {i}")
        print("-" * 80)

        print(
            f"Page: {context['page'] + 1}"
        )

        print(
            f"Distance: {context['distance']:.4f}"
        )

        print("\nText:")
        print(context["text"])


if __name__ == "__main__":
    main()
