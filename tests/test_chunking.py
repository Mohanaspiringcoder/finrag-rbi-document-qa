from langchain_core.documents import Document

from src.chunk_documents import split_documents


def test_split_documents_preserves_page_metadata():
    document = Document(
        page_content="This is a test document. " * 100,
        metadata={"page": 5},
    )

    chunks = split_documents([document])

    assert len(chunks) > 1
    assert all(chunk.metadata["page"] == 5 for chunk in chunks)
