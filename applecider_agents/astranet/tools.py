"""ASTRANet-specific tools: the spectra of a source, the twelve-member classifier and the
object-level novelty stage. The science lives in the astranet package
(`astranet.skyportal.analyze`, the function the SkyPortal analysis service runs); the tools
here fetch the spectra, call it once per object and return rounded numbers.

Spectra come from SkyPortal (SKYPORTAL_URL, SKYPORTAL_TOKEN; read-only requests) or from
local directories of exported files (APPLECIDER_SPECTRA_DIRS, colon-separated, files named
<obj_id>*.dat / .ascii / .txt with two columns). Weights: ASTRANET_MODELS (directory with the
twelve checkpoints) and ASTRANET_NOVELTY (novelty bundle; default <models>/novelty.joblib)."""
from __future__ import annotations
import math, os
from pathlib import Path
from anthropic.lib.tools import beta_tool
from ..core import config as C

_SPECTRA: dict = {}
_RESULT: dict = {}
EXT = (".dat", ".ascii", ".txt")


def _f(x, nd=3):
    try:
        v = float(x)
        return None if not math.isfinite(v) else round(v, nd)
    except (TypeError, ValueError):
        return None


def _local(obj_id):
    out = []
    for d in [Path(p) for p in os.environ.get("APPLECIDER_SPECTRA_DIRS", "").split(":") if p]:
        for p in sorted(d.glob(f"{obj_id}*")):
            if p.suffix.lower() in EXT:
                from astranet.preprocess import read_dat
                w, f = read_dat(str(p))
                out.append(dict(id=p.name, observed_at=None, instrument=None, wavelengths=w.tolist(), fluxes=f.tolist()))
    return out


def _skyportal(obj_id):
    from ..core.skyportal_client import SkyPortal
    sp = SkyPortal()
    rows = sp._req("GET", f"/api/sources/{obj_id}/spectra").get("data", {}).get("spectra", [])
    z = (sp.source(obj_id).get("data") or {}).get("redshift")
    return [dict(id=r.get("id"), observed_at=r.get("observed_at"), instrument=r.get("instrument_name"),
                 wavelengths=r.get("wavelengths") or [], fluxes=r.get("fluxes") or []) for r in rows], z


def spectra(obj_id: str):
    """(list of spectra, source redshift or None, where they came from); cached per process."""
    if obj_id not in _SPECTRA:
        loc = _local(obj_id)
        if loc:
            _SPECTRA[obj_id] = (loc, None, "local")
        elif C.SKYPORTAL_URL and C.SKYPORTAL_TOKEN:
            sp, z = _skyportal(obj_id); _SPECTRA[obj_id] = (sp, z, "skyportal")
        else:
            raise RuntimeError("no spectra source: set SKYPORTAL_URL and SKYPORTAL_TOKEN, or APPLECIDER_SPECTRA_DIRS")
    return _SPECTRA[obj_id]


def analysis(obj_id: str, redshift=None, use_redshift: bool = True) -> dict:
    """One call of astranet.skyportal.analyze per (object, redshift choice); cached per process."""
    sp, z_src, _ = spectra(obj_id)
    z = redshift if redshift is not None else z_src
    key = (obj_id, None if z is None else float(z), bool(use_redshift))
    if key not in _RESULT:
        if not sp:
            raise RuntimeError("the source has no spectrum")
        from astranet.skyportal import analyze
        _RESULT[key] = analyze(sp, z, dict(use_redshift=use_redshift), resource_id=obj_id)
    return _RESULT[key]


@beta_tool
def get_spectra(obj_id: str) -> dict:
    """List the spectra available for the object: how many, and for each one the
    instrument, the observation time, the wavelength range and the number of
    points; also the source redshift when SkyPortal has one. Call it first: the
    ASTRANet tools use every spectrum listed here."""
    try:
        sp, z, src = spectra(obj_id)
        rows = [dict(id=s.get("id"), instrument=s.get("instrument"), observed_at=s.get("observed_at"), n_points=len(s["wavelengths"]),
                     wavelength_min=_f(min(s["wavelengths"]), 0) if s["wavelengths"] else None,
                     wavelength_max=_f(max(s["wavelengths"]), 0) if s["wavelengths"] else None) for s in sp]
        return dict(obj_id=obj_id, n_spectra=len(rows), source=src, redshift=_f(z, 4), spectra=rows,
                    note="ASTRANet reads 3850 to 9000 Angstrom in the observed frame; a spectrum with too little coverage is rejected by the classifier tool.")
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


@beta_tool
def astranet_classify(obj_id: str, redshift: float | None = None, use_redshift: bool = True) -> dict:
    """Run the ASTRANet ensemble (twelve members, two backbones; classes Ia, Ibc,
    SLSN, II, AGN, CV, TDE) on every spectrum of the object. Returns the object
    probabilities (mean over spectra), the argmax, how many spectra agree with
    it, and per spectrum the class, its probability and the fraction of the
    twelve members voting for it. The source redshift is used when known;
    `redshift` overrides it and use_redshift=False withholds it, which shows how
    much the call depends on the redshift."""
    try:
        out = analysis(obj_id, redshift, use_redshift)
        if out.get("status") != "success":
            return dict(error=out.get("message", "ASTRANet failed"), spectra=[dict(id=r.get("id"), status=r.get("status"), reason=r.get("reason")) for r in out.get("results", {}).get("spectra", [])])
        o = out["results"]["object"]
        rows = [dict(id=r.get("id"), instrument=r.get("instrument"), observed_at=r.get("observed_at"), status=r["status"], reason=r.get("reason"),
                     predicted=r.get("predicted"), probability=_f(r.get("probability")), member_agreement=_f(r.get("member_agreement"), 2))
                for r in out["results"]["spectra"]]
        return dict(obj_id=obj_id, predicted=o["predicted"], probability=_f(o["probability"]), probabilities={k: _f(v) for k, v in o["probabilities"].items()},
                    n_spectra=o["n_spectra"], n_rejected=o["n_rejected"], spectra_agreeing=o["spectra_agreeing"],
                    classes_predicted_by_a_spectrum=sorted({r["predicted"] for r in rows if r["status"] == "ok"}),
                    redshift_used=_f(o["redshift_used"], 4), spectra=rows,
                    note="Probabilities are the softmax of the mean member logits; they are not calibrated set guarantees. The service has no spectroscopic phase.")
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


@beta_tool
def astranet_novelty(obj_id: str, redshift: float | None = None, use_redshift: bool = True) -> dict:
    """Object-level novelty from ASTRANet: one score for the object from all its
    spectra (mean embedding, spectral descriptors, LUNA anomaly scores), the
    p-value of that score among ordinary calibration objects of the same
    redshift stratum (redshift known or unknown), and whether it alarms at the
    1, 5, 10 and 15 % alarm rates. Novae are ordinary for this stage."""
    try:
        out = analysis(obj_id, redshift, use_redshift)
        if out.get("status") != "success":
            return dict(error=out.get("message", "ASTRANet failed"))
        n = out["results"]["novelty"]
        if n.get("status") != "ok":
            return dict(obj_id=obj_id, status=n.get("status"), reason=n.get("reason"))
        return dict(obj_id=obj_id, status="ok", p_value=_f(n["p_value"], 4), percentile_among_ordinary=_f(100 * (1 - n["p_value"]), 1),
                    stratum=n["stratum"], n_calibration=n["n_calibration"], alarm=n["alarm"], strictest_alarm_rate=n["alarm_rate"], bundle_version=n.get("version"),
                    note="The p-value is conformal against ordinary objects of the stratum; an alarm rate is the fraction of ordinary objects that alarm, measured at 1.3 / 4.8 / 10.0 / 14.6 % on test objects.")
    except Exception as e:
        return dict(error=f"{type(e).__name__}: {e}")


def astranet_analyze_spectrum(wavelengths: list[float], fluxes: list[float], obj_id: str = "obj", redshift: float | None = None, use_redshift: bool = True) -> dict:
    """Run ASTRANet directly on one spectrum the caller already holds (e.g. from SkyPortal's get_spectra): observed-frame wavelengths in Angstrom and fluxes; optional redshift. Returns the ASTRANet analysis-service result (class probabilities, member votes, novelty p-value and alarms, annotations)."""
    import json
    from astranet.skyportal import analyze
    out = analyze([dict(id=obj_id, wavelengths=wavelengths, fluxes=fluxes)], redshift, dict(use_redshift=use_redshift), resource_id=obj_id)
    return json.loads(json.dumps(out, default=float))


TOOLS = [get_spectra, astranet_classify, astranet_novelty]
