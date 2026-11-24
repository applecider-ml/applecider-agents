"""Offline tests of the anonymised-evaluation helpers (no model, no network)."""
import numpy as np
from applecider_agents.core import blind as B, data as D


def test_anonymous_ids_are_a_shuffled_bijection():
    ids = [f"ZTF20aaaaaa{c}" for c in "abcdefghij"]
    m = B.anonymous_ids(ids, seed=0)
    assert sorted(m) == sorted(ids) and len(set(m.values())) == len(ids)
    assert all(v.startswith("obj_") for v in m.values()) and m == B.anonymous_ids(ids, seed=0)


def test_register_seeds_history_without_epochs():
    hist = dict(found=True, n_detections=7, activity_span_days=12.5, n_per_band={"g": 3, "r": 4},
                first_mjd=60000.1, last_mjd=60012.6)
    B.register("obj_0001", np.empty((0, 15)), dict(ra=10.0, dec=-5.0, host_sep=1.2), history=hist)
    out = D.history("obj_0001")
    assert out == dict(found=True, n_detections=7, activity_span_days=12.5, n_per_band={"g": 3, "r": 4})
    ctx = D.context("obj_0001")
    assert "ra" not in ctx and "dec" not in ctx and ctx["host_sep"] == 1.2
    # a later plain seed drops the seeded history
    D.seed("obj_0001", np.empty((0, 15)), {}, metadata={"photometry_rows": []})
    assert D.history("obj_0001") == dict(found=False, source="provided")
