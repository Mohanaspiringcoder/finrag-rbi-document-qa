import sys
from pathlib import Path

import streamlit as st


# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieve import (
    load_embedding_model,
    load_vector_db,
    retrieve_documents,
)

from src.generate_answer import generate_answer


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RBI Document Q&A",
    page_icon="🏦",
)


# ---------------------------------------------------------------------------
# Load RAG components
# ---------------------------------------------------------------------------

@st.cache_resource
def load_rag_components():
    """Load the embedding model and ChromaDB once."""

    embedding_model = load_embedding_model()
    collection = load_vector_db()

    return embedding_model, collection


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def main():
    st.title("🏦 RBI Document Q&A")

    st.write(
        "Ask questions about the RBI document and get answers "
        "based only on the retrieved RBI context."
    )

    question = st.text_input(
        "Ask an RBI question",
        placeholder="Example: What is the monthly PPI limit?",
    )

    if st.button("Ask Question", type="primary"):

        if not question.strip():
            st.warning("Please enter a question.")
            return

        # Load RAG components.
        with st.spinner("Searching the RBI document..."):
            embedding_model, collection = load_rag_components()

            contexts = retrieve_documents(
                question,
                embedding_model,
                collection,
                top_k=1,
            )

        # No relevant information.
        if not contexts:
            st.warning(
                "I could not find the answer in the provided RBI document."
            )
            return

        # Generate answer.
        with st.spinner("Generating answer..."):
            answer = generate_answer(
                question,
                contexts,
            )

        # Display answer.
        st.subheader("Answer")
        st.write(answer)

        # Display source.
        st.subheader("Source")

        for context in contexts:
            st.write(
                f"**RBI Page {context['page'] + 1}**"
            )

            st.caption(
                f"Retrieval distance: {context['distance']:.4f}"
            )


if __name__ == "__main__":
    main()