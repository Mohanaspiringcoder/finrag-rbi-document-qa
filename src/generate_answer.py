import ollama

from src.config import LLM_MODEL, TOP_K
from src.retrieve import (
    load_embedding_model,
    load_vector_db,
    retrieve_documents,
)


def build_prompt(question, contexts):
    """Build the prompt for the local LLM."""

    context_text = "\n\n".join(
        [
            f"RBI Page {item['page'] + 1}:\n{item['text']}"
            for item in contexts
        ]
    )

    return f"""You are an RBI document question-answering assistant.

Answer the question using ONLY the RBI text provided below.

RBI TEXT:
{context_text}

QUESTION:
{question}

Instructions:
- Use only information from the RBI text.
- Do not use outside knowledge.
- Do not invent information.
- Do not repeat unrelated information.
- Keep the answer concise.

ANSWER:"""


def generate_answer(question, contexts):
    """Generate an answer using the local Llama model."""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": build_prompt(
                    question,
                    contexts,
                ),
            }
        ],
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"].strip()


def is_failed_answer(answer):
    """Check whether the model failed to answer."""

    failure_phrases = [
        "i could not find the answer",
        "could not find the answer",
        "answer is not present",
        "not found in the provided",
        "cannot find the answer",
        "unable to find the answer",
    ]

    answer_lower = answer.lower()

    return any(
        phrase in answer_lower
        for phrase in failure_phrases
    )


def generate_fallback_answer(question, contexts):
    """Extract an answer directly from the retrieved RBI text."""

    context_text = "\n\n".join(
        [
            f"RBI Page {item['page'] + 1}:\n{item['text']}"
            for item in contexts
        ]
    )

    prompt = f"""Extract the answer from the RBI text.

RBI TEXT:
{context_text}

QUESTION:
{question}

Return only the answer.
Use only information explicitly stated in the RBI text.
Keep the answer concise.

ANSWER:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"].strip()


def main():
    print("=" * 80)
    print("RBI DOCUMENT Q&A")
    print("=" * 80)

    embedding_model = load_embedding_model()
    collection = load_vector_db()

    question = input(
        "\nAsk an RBI question: "
    ).strip()

    if not question:
        print("\nPlease enter a question.")
        return

    contexts = retrieve_documents(
        question,
        embedding_model,
        collection,
        top_k=TOP_K,
    )

    if not contexts:
        print("\nANSWER:")
        print(
            "I could not find the answer "
            "in the provided RBI document."
        )
        return

    print("\nRetrieved context:")
    print("-" * 80)

    for context in contexts:
        print(
            f"\nRBI Page {context['page'] + 1}"
        )

        print(
            f"Distance: "
            f"{context['distance']:.4f}"
        )

        #print(
          #  context["text"][:500]
        #)

    print("\nGenerating answer...")
    print("-" * 80)

    answer = generate_answer(
        question,
        contexts,
    )

    if is_failed_answer(answer):
        answer = generate_fallback_answer(
            question,
            contexts,
        )

    print("\nANSWER:")
    print(answer)


if __name__ == "__main__":
    main()
