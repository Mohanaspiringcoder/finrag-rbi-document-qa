from pathlib import Path

import pymupdf
from langchain_core.documents import Document


PDF_PATH = Path("data/documents/rbi_prepaid_payment_instruments.pdf")


def load_documents():
    """Load the RBI PDF and return one Document per page."""

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

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


if __name__ == "__main__":
    documents = load_documents()

    print(f"Number of pages loaded: {len(documents)}")

    for document in documents[:3]:
        print("\n" + "=" * 80)
        print(f"PAGE {document.metadata['page'] + 1}")
        print("=" * 80)

        print("\nMetadata:")
        print(document.metadata)

        print("\nText preview:")
        print(document.page_content[:1000])
