from conflict_eval.data.sampling import (
    assign_margin_bin,
    compute_margin_bin_edges,
    sample_balanced_across_bins,
    sample_candidates,
)


def test_sample_candidates_is_deterministic():
    items = [{"id": str(i)} for i in range(50)]
    a = sample_candidates(items, 10, seed=7)
    b = sample_candidates(items, 10, seed=7)
    assert a == b
    assert len(a) == 10


def test_sample_candidates_returns_all_if_n_exceeds_pool():
    items = [{"id": str(i)} for i in range(5)]
    assert len(sample_candidates(items, 100, seed=1)) == 5


def test_margin_bin_edges_and_assignment():
    margins = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
    edges = compute_margin_bin_edges(margins, n_bins=3)
    assert len(edges) == 2
    assert assign_margin_bin(margins[0], edges) == "low"
    assert assign_margin_bin(margins[-1], edges) == "high"


def test_sample_balanced_across_bins_spreads_across_bins():
    items = (
        [{"id": f"low-{i}", "margin_bin": "low"} for i in range(10)]
        + [{"id": f"med-{i}", "margin_bin": "medium"} for i in range(10)]
        + [{"id": f"high-{i}", "margin_bin": "high"} for i in range(10)]
    )
    sampled = sample_balanced_across_bins(items, target_n=9, seed=3)
    assert len(sampled) == 9
    bins_present = {item["margin_bin"] for item in sampled}
    assert bins_present == {"low", "medium", "high"}


def test_sample_balanced_across_bins_never_exceeds_available_items():
    items = [{"id": "1", "margin_bin": "low"}]
    sampled = sample_balanced_across_bins(items, target_n=10, seed=3)
    assert len(sampled) == 1
