"""Unit tests for agentic/tools/embeddings.py.

No network/API key required: these test the math and the injection point
(`embed_fn`) directly, never the real OpenAI client.
"""
from agentic.tools.embeddings import cosine_similarity, embed_texts, rank_by_similarity


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_similarity_zero_vector_does_not_raise():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_rank_by_similarity_orders_highest_first_and_respects_top_k():
    query = [1, 0]
    candidates = [
        ("orthogonal", [0, 1]),
        ("identical", [1, 0]),
        ("diagonal", [1, 1]),
    ]

    ranked = rank_by_similarity(query, candidates, top_k=2)

    assert [item for item, _ in ranked] == ["identical", "diagonal"]
    assert ranked[0][1] == 1.0


def test_embed_texts_empty_input_returns_empty_list():
    assert embed_texts([]) == []


def test_embed_texts_uses_injected_embed_fn_without_touching_default():
    calls = []

    def fake_embed_fn(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    result = embed_texts(["hello", "world"], embed_fn=fake_embed_fn)

    assert result == [[1.0, 0.0], [1.0, 0.0]]
    assert calls == [["hello", "world"]]
