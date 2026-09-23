"""Minimal SkyPortal REST client. Nothing here runs unless FLARE_MODE allows it."""
from __future__ import annotations
import json, urllib.request
from . import config as C


class SkyPortal:
    def __init__(self, url: str = C.SKYPORTAL_URL, token: str = C.SKYPORTAL_TOKEN):
        if not url or not token:
            raise RuntimeError("SKYPORTAL_URL and SKYPORTAL_TOKEN must be set")
        self.url, self.token = url.rstrip("/"), token

    def _req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Authorization": f"token {self.token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)

    def source(self, obj_id):
        return self._req("GET", f"/api/sources/{obj_id}")

    def recent_sources(self, saved_after_iso: str, group_ids=None, n=200):
        q = f"/api/sources?savedAfter={saved_after_iso}&numPerPage={n}" + (f"&group_ids={','.join(map(str, group_ids))}" if group_ids else "")
        return self._req("GET", q).get("data", {}).get("sources", [])

    def comment(self, obj_id, text, group_ids=None):
        return self._req("POST", f"/api/sources/{obj_id}/comments", dict(text=text, group_ids=group_ids or C.SKYPORTAL_GROUP_IDS or None))

    def annotation(self, obj_id, data: dict, origin="FLARE", group_ids=None):
        return self._req("POST", f"/api/sources/{obj_id}/annotations", dict(origin=origin, data=data, group_ids=group_ids or C.SKYPORTAL_GROUP_IDS or None))


def publish(report, mode: str = C.MODE):
    """Post according to the mode: dry-run logs only; comment posts a comment and an annotation."""
    if mode == "dry-run":
        return dict(posted=False, mode=mode, would_post=report.comment())
    if mode in ("comment", "analysis"):
        sp = SkyPortal()
        c = sp.comment(report.obj_id, report.comment())
        a = sp.annotation(report.obj_id, dict(verdict=report.verdict, priority=report.priority, label=report.flare_label,
                                              prediction_set=report.prediction_set, credibility=report.credibility, energy_percentile=report.energy_percentile))
        return dict(posted=True, mode=mode, comment=c.get("status"), annotation=a.get("status"))
    raise ValueError(f"unknown FLARE_MODE {mode}")
