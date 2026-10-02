"""Gate on the REAL built KG: leakage masking and splits hold on actual data. Skipped if the KG has not been built locally."""
import pytest

from ayurveda_kg.kg import load_tables
from ayurveda_kg.labels import CYP5, silver_labels
from ayurveda_kg.masking import assert_no_label_leak, mask_label_edges
from ayurveda_kg.splits import assert_disjoint, assign_folds, herb_group_folds

pytestmark = pytest.mark.skipif(not __import__("pathlib").Path("data/processed/kg/nodes_Compound.parquet").exists(),
                                reason="KG not built locally (IMPPAT-derived data is not committed)")


@pytest.fixture(scope="module")
def kg():
    return load_tables("data/processed/kg")


def test_real_masked_graph_has_no_label_path_and_cannot_rederive_labels(kg):
    nodes, edges = kg
    masked = mask_label_edges(edges)
    assert_no_label_leak(masked)
    assert (silver_labels(nodes, edges).label == 1).sum() > 5000              # sanity: unmasked graph does yield many positives
    assert (silver_labels(nodes, masked).label == 1).sum() == 0               # masked graph yields none
    cyp = {f"gene:{g}" for g in CYP5}
    assert not any(masked[t].dst.isin(cyp).any() for t in ("modulates", "targets"))
    assert len(masked["contains"]) == len(edges["contains"])                  # structure the model may use is intact


def test_real_cold_compound_folds_partition_all_compounds(kg):
    nodes, _ = kg
    folds = assign_folds(nodes["Compound"]["id"], k=5, seed=0)
    assert len(folds) == len(nodes["Compound"]) and set(folds.values()) == set(range(5))


def test_real_herb_folds_are_leak_free_and_every_herb_held_out_once(kg):
    nodes, edges = kg
    held = []
    for test_herbs, train, test, shared in herb_group_folds(edges["contains"], k=5, seed=0):
        assert_disjoint(train, test); assert_disjoint(train, shared); assert_disjoint(test, shared)
        assert len(train) > 0 and len(test) > 0
        held += sorted(test_herbs)
    assert sorted(held) == sorted(nodes["Herb"]["id"])
