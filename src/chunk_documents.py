from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.load_documents import load_documents


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_documents(documents):
    """Split pages into overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.split_documents(documents)


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):
        print("\n" + "=" * 80)
        print(f"CHUNK {i + 1}")
        print("=" * 80)

        print("\nMetadata:")
        print(chunk.metadata)

        print("\nCharacters:")
        print(len(chunk.page_content))

        print("\nText:")
        print(chunk.page_content)
