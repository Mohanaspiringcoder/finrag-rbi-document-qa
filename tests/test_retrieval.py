from src.retrieve import retrieve_documents


import numpy as np


class FakeEmbeddingModel:
    def encode(self, questions):
        return np.array([[0.1, 0.2, 0.3]])


class FakeCollection:
    def query(self, query_embeddings, n_results):
        return {
            "documents": [["relevant text", "irrelevant text"]],
            "metadatas": [[
                {"page": 8},
                {"page": 15},
            ]],
            "distances": [[0.5, 1.5]],
        }


def test_retrieve_documents_filters_by_distance():
    contexts = retrieve_documents(
        question="What is the PPI limit?",
        embedding_model=FakeEmbeddingModel(),
        collection=FakeCollection(),
        top_k=2,
        distance_threshold=1.0,
    )

    assert len(contexts) == 1
    assert contexts[0]["text"] == "relevant text"
    assert contexts[0]["page"] == 8
    assert contexts[0]["distance"] == 0.5
