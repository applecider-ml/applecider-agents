"""Offline SkyPortal API contract, ingestion and cache-refresh regression tests."""
import io
import json
from urllib.parse import urlparse, parse_qs
from types import SimpleNamespace
import numpy as np
import pytest
from applecider_agents.core.skyportal_client import SkyPortal, publish
from applecider_agents.core import data as D
from applecider_agents.astranet import tools as A
from applecider_agents.flare import tools as F
from applecider_agents.skyportal_connector import workflow as W


@pytest.fixture(autouse=True)
def isolated_caches():
    yield
    D._SEEDED.clear()
    D._SEED_METADATA.clear()
    A._SPECTRA.clear()
    A._RESULT.clear()
    F._CACHE.clear()
    F._RESULTS.clear()


class FakeSky:
    def source(self, oid):
        return {"data": {"id": oid, "ra": 42., "dec": 12., "redshift": .04}}

    def photometry(self, oid):
        return [dict(mjd=60000+i, filter="ztfg" if i % 2 else "ztfr", mag=19-i*.1, magerr=.1) for i in range(12)]

    def spectra(self, oid):
        return [dict(id=1, wavelengths=[4000, 5000, 6000], fluxes=[1., 2., 1.])]

    def recent_sources(self, *args, **kwargs):
        return [{"id": "bad"}, {"id": "good"}]


def test_photometry_filters_limits_invalid_and_dedup():
    p = dict(mjd=60000, filter="ztfg", mag=19, magerr=.1)
    rows, excluded = W.photometry_rows([p, p, dict(p, filter="sdssg"), dict(p, mag=None), dict(p, magerr=0), dict(p, mjd=float("nan"))])
    assert rows == [(60000., 1, 19., .1)]
    assert excluded == dict(unsupported_filter=1, upper_limit_or_invalid=3, duplicate=1)


def test_snapshot_seeds_both_and_invalidates_old_analysis():
    F._CACHE[("X", 100.)] = "old"
    F._RESULTS[("X", 100.)] = "old"
    A._RESULT[("X", .04, True)] = "old"
    provenance = W.ingest("X", FakeSky())
    assert provenance["n_photometry_used"] == 12
    assert len(provenance["snapshot_sha256"]) == 64
    assert D.coords("X") == (42., 12.)
    assert len(D.events("X", 5)) == 6
    assert D.history("X")["activity_span_days"] == 11
    assert A.spectra("X")[1] == .04
    assert not F._CACHE and not F._RESULTS and not A._RESULT
    assert "z_phot" not in D.context("X")
    assert D.context("X")["_context_missing"]


def test_full_history_not_cut_at_100_days():
    client = FakeSky()
    client.photometry = lambda oid: [dict(mjd=mjd, filter="ztfg", mag=19, magerr=.1) for mjd in [60000, 60400]]
    W.ingest("X", client)
    assert len(D.events("X", 500)) == 2
    assert D.history("X")["activity_span_days"] == 400


def test_failed_spectra_do_not_reuse_previous_spectrum():
    A.seed_spectra("X", [dict(id="old")], .1)
    client = FakeSky()
    def fail(oid):
        raise RuntimeError("spectra not accessible")
    client.spectra = fail
    result = W.ingest("X", client)
    assert result["fetch_status"]["spectra_fetch"] == "error"
    assert A.spectra("X")[0] == []
    assert len(D.events("X")) == 12


def test_empty_photometry_does_not_reuse_previous_curve():
    W.ingest("X", FakeSky())
    client = FakeSky()
    client.photometry = lambda oid: []
    W.ingest("X", client)
    assert len(D.events("X")) == 0
    assert not D.history("X")["found"]
    assert A.spectra("X")[0]


def test_http_contract_auth_query_encoding_and_spectrum_normalization(monkeypatch):
    requests = []
    def open_(request, timeout):
        requests.append(request)
        if "/photometry?" in request.full_url:
            data = [dict(mjd=60000, filter="ztfg", mag=19, magerr=.1)]
        elif request.full_url.endswith("/spectra"):
            data = dict(spectra=[dict(id=2, instrument={"name": "test"}, wavelengths=[4000], fluxes=[1])])
        else:
            data = dict(id="X/Y")
        return io.BytesIO(json.dumps(dict(status="success", data=data)).encode())
    monkeypatch.setattr("urllib.request.urlopen", open_)
    client = SkyPortal("https://example.invalid", "test-token")
    assert client.source("X/Y")["data"]["id"] == "X/Y"
    assert client.photometry("X/Y")[0]["mag"] == 19
    assert client.spectra("X/Y")[0]["instrument"] == "test"
    assert all("X%2FY" in r.full_url for r in requests)
    assert all(r.get_header("Authorization") == "token test-token" for r in requests)
    assert parse_qs(urlparse(requests[1].full_url).query) == {"format": ["mag"], "magsys": ["ab"]}


def test_api_error_not_treated_as_empty_success(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b'{"status":"error","message":"denied"}'))
    with pytest.raises(RuntimeError, match="denied"):
        SkyPortal("https://example.invalid", "token").source("X")


def test_recent_sources_pagination(monkeypatch):
    client = SkyPortal("https://example.invalid", "token")
    pages = []
    def req(method, path):
        params = parse_qs(urlparse(path).query)
        page = int(params["pageNumber"][0])
        pages.append(page)
        assert params["savedAfter"] == ["2026-01-01T00:00:00+00:00"]
        return {"data": {"sources": [{"id": str(i)} for i in (range(100) if page == 1 else range(100, 105))], "totalMatches": 105}}
    monkeypatch.setattr(client, "_req", req)
    assert len(client.recent_sources("2026-01-01T00:00:00+00:00", n=103)) == 103
    assert pages == [1, 2]


def test_scan_records_per_source_errors(monkeypatch):
    def investigate(oid, **kwargs):
        if oid == "bad":
            raise ValueError("missing data")
        return SimpleNamespace(priority=1, discovery_potential=.2, model_dump=lambda: {"obj_id": oid})
    monkeypatch.setattr(W, "investigate", investigate)
    result = W.scan("2026-01-01", client=FakeSky())
    assert result["reports"] == [{"obj_id": "good"}]
    assert result["errors"][0]["obj_id"] == "bad"


def test_publish_default_dry_run_and_joint_origin(monkeypatch):
    from applecider_agents.core import config as C
    monkeypatch.setattr(C, "MODE", "dry-run")
    report = SimpleNamespace(obj_id="X", ORIGIN="AppleCiDEr-collaboration", comment=lambda: "review", annotation=lambda: {"models": "joint"})
    client = SimpleNamespace()
    assert not publish(report, client=client)["posted"]
    sent = []
    client.comment = lambda *args: sent.append(args) or {"status": "success"}
    client.annotation = lambda *args, **kwargs: sent.append(kwargs) or {"status": "success"}
    assert publish(report, mode="comment", client=client)["posted"]
    assert sent[-1]["origin"] == "AppleCiDEr-collaboration"
