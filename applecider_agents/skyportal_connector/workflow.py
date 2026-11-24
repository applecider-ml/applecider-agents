"""SkyPortal ingestion for FLARE photometry and ASTRANet spectra.

Uses only ZTF filters supported by the trained FLARE model. Other passbands and
upper limits are counted and excluded, never silently re-labelled as ZTF.
"""
from __future__ import annotations
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np
from ..core.skyportal_client import SkyPortal
from ..core import data as DATA
from ..astranet.tools import seed_spectra
from ..collaboration.workflow import collaborate

FILTERS = {"ztfg": 1, "ztfr": 2, "ztfi": 3}


def photometry_rows(points):
    rows, seen = [], set()
    excluded = dict(unsupported_filter=0, upper_limit_or_invalid=0, duplicate=0)
    for point in points:
        fid = FILTERS.get(str(point.get("filter", "")).lower())
        if fid is None:
            excluded["unsupported_filter"] += 1
            continue
        try:
            t, mag, err = (float(point[k]) for k in ("mjd", "mag", "magerr"))
            if not all(math.isfinite(x) for x in (t, mag, err)) or err <= 0:
                raise ValueError("invalid photometry")
        except (KeyError, TypeError, ValueError):
            excluded["upper_limit_or_invalid"] += 1
            continue
        key = (t, fid, mag, err)
        if key in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(key)
        rows.append(key)
    return sorted(rows), excluded


def ingest(obj_id, client=None):
    """Replace cached model inputs with one current, attributable source snapshot."""
    from flare.fetch import to_events
    client = client or SkyPortal()
    source = client.source(obj_id).get("data")
    if not isinstance(source, dict):
        raise ValueError("SkyPortal source response must contain an object")
    warnings, status = [], {}
    try:
        phot = client.photometry(obj_id)
        status["photometry_fetch"] = "ok"
    except Exception as exc:
        phot = []
        status["photometry_fetch"] = "error"
        warnings.append(f"SkyPortal photometry unavailable: {type(exc).__name__}: {exc}")
    try:
        spectra = client.spectra(obj_id)
        status["spectra_fetch"] = "ok"
    except Exception as exc:
        spectra = []
        status["spectra_fetch"] = "error"
        warnings.append(f"SkyPortal spectra unavailable: {type(exc).__name__}: {exc}")
    rows, excluded = photometry_rows(phot)
    if any(excluded.values()):
        warnings.append(f"Photometry excluded from FLARE inputs: {excluded}")
    coordinates = None
    try:
        ra, dec = float(source["ra"]), float(source["dec"])
        if math.isfinite(ra) and math.isfinite(dec):
            coordinates = (ra, dec)
    except (KeyError, ValueError, TypeError):
        pass
    # Source redshift is for ASTRANet. It is not a measured host photo-z.
    context = {"_source": "skyportal", "_context_missing": True}
    warnings.append("SkyPortal snapshot does not supply host/Gaia/WISE features; missing context is not evidence of hostlessness.")
    events = to_events(rows, horizon_days=float("inf")) if rows else np.empty((0, 15))
    DATA.seed(obj_id, events, context, metadata=dict(photometry_rows=rows, coordinates=coordinates, catalog={}))
    seed_spectra(obj_id, spectra, source.get("redshift"), source="skyportal")
    snapshot = dict(source=source, photometry=phot, spectra=spectra)
    return dict(connector="skyportal", obj_id=obj_id, fetched_at=datetime.now(timezone.utc).isoformat(),
                snapshot_sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode()).hexdigest(),
                n_photometry_used=len(rows), n_spectra=len(spectra), excluded_photometry=excluded,
                fetch_status=status, warnings=warnings,
                source_classifications=source.get("classifications", []))


def investigate(obj_id, *, client=None, model="joint", horizon_days=100.0, use_llm=False):
    client = client or SkyPortal()
    provenance = ingest(obj_id, client)
    return collaborate(obj_id, model=model, horizon_days=horizon_days, use_llm=use_llm, provenance=provenance)


def scan(saved_after, *, client=None, group_ids=None, limit=20, model="joint", horizon_days=100.0, use_llm=False):
    """Read-only batch; one bad source does not discard the rest of the run."""
    client = client or SkyPortal()
    reports, errors = [], []
    for source in client.recent_sources(saved_after, group_ids=group_ids, n=limit):
        try:
            reports.append(investigate(source["id"], client=client, model=model,
                                       horizon_days=horizon_days, use_llm=use_llm))
        except Exception as exc:
            errors.append(dict(obj_id=source["id"], error=f"{type(exc).__name__}: {exc}"))
    reports.sort(key=lambda r: (r.priority, r.discovery_potential), reverse=True)
    return dict(reports=[r.model_dump() for r in reports], errors=errors)
