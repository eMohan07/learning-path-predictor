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
def test_milestone_3():
    from app.core import milestone_3, predict_next_courses, COURSE_CATALOG, get_default_model

    model = get_default_model()
    result = milestone_3(model=model)

    assert isinstance(result, dict)
    assert "evaluations" in result
    evals = result["evaluations"]
    assert len(evals) >= 4

    # Test individual prediction calls
    ds_history = ["CS101", "PY101", "MATH101", "DS201", "STAT201"]
    recs = predict_next_courses(ds_history, top_n=3, model=model)
    assert len(recs) <= 3
    assert len(recs) > 0

    top_rec = recs[0]
    assert "course_id" in top_rec
    assert "title" in top_rec
    assert "track" in top_rec
    assert "probability" in top_rec
    assert "confidence_pct" in top_rec
    assert "rationale" in top_rec

    # None of the completed courses should be recommended
    rec_ids = [r["course_id"] for r in recs]
    for cid in ds_history:
        assert cid not in rec_ids

    # For students with MATH101 + DS201 + STAT201, ML301 is the quintessential next step
    assert "ML301" in rec_ids

    # Prerequisites must be satisfied for all recommendations
    for r in recs:
        prereqs = COURSE_CATALOG[r["course_id"]]["prerequisites"]
        for p in prereqs:
            assert p in ds_history


# Milestone 4: Create a Streamlit interface for students to input their completed courses and view recommendations.
def test_milestone_4():
    import py_compile
    from app.core import milestone_4

    # Ensure main app file is syntactically valid and clean
    py_compile.compile("app/main.py", doraise=True)

    result = milestone_4(sample_completed=["CS101", "PY101"], top_n=4)
    assert isinstance(result, dict)
    assert result["status"] == "ready"
    assert result["top_n"] == 4
    assert len(result["recommendations"]) == 4

    for rec in result["recommendations"]:
        assert "course_id" in rec
        assert "title" in rec
        assert "track" in rec
        assert "relative_score" in rec
        assert "confidence_pct" in rec
        assert "rationale" in rec


# Milestone 5: Analyze which course "itemsets" the model has learned.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

