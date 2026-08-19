import shutil
from pathlib import Path

import chromadb
import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


PDF_PATH = Path("data/documents/rbi_prepaid_payment_instruments.pdf")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZES = [500, 1000, 1500]
OVERLAP_RATIO = 0.2

QUESTIONS = [
    {
        "question": "What is the maximum amount that can be loaded into such PPIs during a month?",
        "expected_page": 9,
    },
    {
        "question": "What is the maximum amount that can remain outstanding in such PPIs at any point in time?",
        "expected_page": 9,
    },
    {
        "question": "What is the monthly limit for funds transfer from such PPIs?",
        "expected_page": 10,
    },
    {
        "question": "What is the maximum cash withdrawal limit for non-bank issued PPIs?",
        "expected_page": 11,
    },
    {
        "question": "What does the term PPI Holder mean?",
        "expected_page": 2,
    },
]


def load_documents():
    pdf = pymupdf.open(PDF_PATH)

    documents = []

    for page_number, page in enumerate(pdf):
        documents.append(
            Document(
                page_content=page.get_text(),
                metadata={
                    "source": str(PDF_PATH),
                    "page": page_number,
                    "total_pages": len(pdf),
                },
            )
        )

    pdf.close()

    return documents


def create_chunks(documents, chunk_size):
    chunk_overlap = int(chunk_size * OVERLAP_RATIO)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_documents(documents)


def evaluate_collection(collection, model):
    recall_at_1 = 0
    recall_at_5 = 0
    reciprocal_ranks = []

    for item in QUESTIONS:

        query_embedding = model.encode(
            [item["question"]]
        )[0]

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,
        )

        retrieved_pages = [
            metadata["page"]
            for metadata in results["metadatas"][0]
        ]

        expected_page = item["expected_page"]

        # Recall@1
        if retrieved_pages[0] == expected_page:
            recall_at_1 += 1

        # Recall@5 + MRR
        rank = None

        for position, page in enumerate(
            retrieved_pages,
            start=1,
        ):
            if page == expected_page:
                rank = position
                break

        if rank is not None:
            recall_at_5 += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    total = len(QUESTIONS)

    return {
        "recall_at_1": recall_at_1 / total,
        "recall_at_5": recall_at_5 / total,
        "mrr": sum(reciprocal_ranks) / total,
    }


def main():

    print("=" * 80)
    print("CHUNKING EXPERIMENT")
    print("=" * 80)

    documents = load_documents()

    print(f"\nPages loaded: {len(documents)}")

    model = SentenceTransformer(MODEL_NAME)

    results = []

    for chunk_size in CHUNK_SIZES:

        db_path = f"chroma_db_{chunk_size}"

        # Remove previous experiment database if it exists
        if Path(db_path).exists():
            shutil.rmtree(db_path)

        chunks = create_chunks(
            documents,
            chunk_size,
        )

        print("\n" + "-" * 80)
        print(f"Chunk size: {chunk_size}")
        print(f"Chunk overlap: {int(chunk_size * OVERLAP_RATIO)}")
        print(f"Number of chunks: {len(chunks)}")

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        embeddings = model.encode(
            texts,
            show_progress_bar=True,
        )

        client = chromadb.PersistentClient(
            path=db_path
        )

        collection = client.get_or_create_collection(
            name=f"rbi_{chunk_size}"
        )

        collection.upsert(
            ids=[
                f"chunk_{i}"
                for i in range(len(chunks))
            ],
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[
                chunk.metadata
                for chunk in chunks
            ],
        )

        metrics = evaluate_collection(
            collection,
            model,
        )

        results.append(
            (
                chunk_size,
                len(chunks),
                metrics,
            )
        )

    print("\n")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(
        f"{'Chunk':<10}"
        f"{'Chunks':<10}"
        f"{'Recall@1':<12}"
        f"{'Recall@5':<12}"
        f"{'MRR':<10}"
    )

    for chunk_size, number_of_chunks, metrics in results:

        print(
            f"{chunk_size:<10}"
            f"{number_of_chunks:<10}"
            f"{metrics['recall_at_1']:.2%}       "
            f"{metrics['recall_at_5']:.2%}       "
            f"{metrics['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()
