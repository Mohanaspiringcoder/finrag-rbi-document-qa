import chromadb
from sentence_transformers import SentenceTransformer

from src.chunk_documents import split_documents
from src.config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from src.load_documents import load_documents


def build_vector_database():
    """Build the Chroma vector database from the RBI document."""

    # Load PDF pages.
    documents = load_documents()

    # Split pages into chunks.
    chunks = split_documents(documents)

    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    # Load embedding model.
    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Extract chunk text.
    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # Generate embeddings.
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
    )

    print(
        f"Embeddings shape: {embeddings.shape}"
    )

    # Create persistent Chroma database.
    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Create IDs for chunks.
    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    # Store chunks, embeddings and metadata.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            chunk.metadata
            for chunk in chunks
        ],
    )

    print("\nVector database built successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(
        f"Stored chunks: {collection.count()}"
    )


if __name__ == "__main__":
    build_vector_database()
