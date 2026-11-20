"""Your tests. Each milestone in README.md gets one here: write the test first, then make it pass.

They start skipped. Remove the `skip` line as you work on each milestone.
"""
import pytest

# Milestone 1: Generate mock student course completion data as sequences of course IDs.
def test_milestone_1():
    from app.core import milestone_1, generate_student_sequences, COURSE_CATALOG

    result = milestone_1(n_students=50, seed=123)
    assert isinstance(result, dict)
    assert "sequences" in result
    assert "catalog" in result
    assert "summary" in result

    sequences = result["sequences"]
    assert len(sequences) == 50
    assert all(isinstance(seq, list) for seq in sequences)

    # Check that each sequence has reasonable length and valid course IDs
    for seq in sequences:
        assert len(seq) >= 3
        assert len(seq) <= 8
        for cid in seq:
            assert cid in COURSE_CATALOG

    # Verify prerequisite logic: introductory courses appear before advanced ones
    for seq in sequences:
        seen = set()
        for cid in seq:
            prereqs = COURSE_CATALOG[cid]["prerequisites"]
            for p in prereqs:
                assert p in seen, f"Prerequisite {p} not satisfied before {cid} in sequence {seq}"
            seen.add(cid)

    # Check summary metrics
    summary = result["summary"]
    assert summary["n_students"] == 50
    assert summary["min_sequence_length"] >= 3
    assert summary["max_sequence_length"] <= 8
    assert summary["unique_courses"] > 10


# Milestone 2: Pretrain a small language model on these course ID sequences.
def test_milestone_2():
    from app.core import milestone_2, CourseLanguageModel

    result = milestone_2(epochs=10, seed=42)
    assert isinstance(result, dict)
    assert "model" in result
    assert "history" in result
    assert "initial_loss" in result
    assert "final_loss" in result
    assert "val_perplexity" in result

    model = result["model"]
    assert isinstance(model, CourseLanguageModel)
    assert model.vocab_size > 20

    # Ensure model learned: loss should decrease
    assert result["final_loss"] < result["initial_loss"]
    assert result["val_perplexity"] > 0
    assert result["val_perplexity"] < model.vocab_size

    # Test probability distribution output
    probs = model.predict_next_probabilities(["CS101", "PY101"])
    assert isinstance(probs, dict)
    assert "CS101" not in probs  # Already completed courses excluded
    assert "PY101" not in probs
    assert abs(sum(probs.values()) - 1.0) < 1e-4


# Milestone 3: Implement a function to predict the next likely course given a partial sequence.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Create a Streamlit interface for students to input their completed courses and view recommendations.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Analyze which course "itemsets" the model has learned.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

