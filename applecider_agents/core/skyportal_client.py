"""SkyPortal REST client. Reads are independent of the explicit publishing gate."""
from __future__ import annotations
import json, urllib.request
from urllib.parse import quote, urlencode
from . import config as C


class SkyPortal:
    def __init__(self, url: str | None = None, token: str | None = None):
        url = C.SKYPORTAL_URL if url is None else url
        token = C.SKYPORTAL_TOKEN if token is None else token
        if not url or not token:
            raise RuntimeError("SKYPORTAL_URL and SKYPORTAL_TOKEN must be set")
        self.url, self.token = url.rstrip("/"), token

    def _req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Authorization": f"token {self.token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.load(r)
        if not isinstance(result, dict) or result.get("status") != "success":
            raise RuntimeError(f"SkyPortal {method} request failed: {result.get('message', 'invalid response') if isinstance(result, dict) else 'invalid response'}")
        return result

    def source(self, obj_id):
        return self._req("GET", f"/api/sources/{quote(str(obj_id), safe='')}")

    def photometry(self, obj_id):
        """AB magnitudes, preserving upper limits for the adapter to account for."""
        data = self._req("GET", f"/api/sources/{quote(str(obj_id), safe='')}/photometry?format=mag&magsys=ab").get("data")
        if not isinstance(data, list):
            raise ValueError("SkyPortal photometry response must contain a data list")
        return data

    def spectra(self, obj_id):
        data = self._req("GET", f"/api/sources/{quote(str(obj_id), safe='')}/spectra").get("data", {})
        rows = data.get("spectra")
        if not isinstance(rows, list):
            raise ValueError("SkyPortal spectra response must contain a spectra list")
        return [dict(id=r.get("id"), observed_at=r.get("observed_at"),
                     instrument=r.get("instrument_name") or (r.get("instrument") or {}).get("name"),
                     wavelengths=r.get("wavelengths") or [], fluxes=r.get("fluxes") or []) for r in rows]

    def recent_sources(self, saved_after_iso: str, group_ids=None, n=200):
        """Read up to n unique sources, following SkyPortal's numbered pages."""
        if n <= 0:
            return []
        found, seen, page = [], set(), 1
        page_size = min(n, 100)
        while len(found) < n:
            params = dict(savedAfter=saved_after_iso, numPerPage=page_size, pageNumber=page)
            if group_ids:
                params["group_ids"] = ",".join(map(str, group_ids))
            data = self._req("GET", "/api/sources?" + urlencode(params)).get("data", {})
            rows = data.get("sources")
            if not isinstance(rows, list):
                raise ValueError("SkyPortal sources response must contain a sources list")
            previous = len(found)
            for row in rows:
                if row["id"] not in seen:
                    found.append(row)
                    seen.add(row["id"])
            if not rows or len(rows) < page_size or len(found) == previous:
                break
            total = data.get("totalMatches")
            if total is not None and page * page_size >= int(total):
                break
            page += 1
        return found[:n]

    def comment(self, obj_id, text, group_ids=None):
        return self._req("POST", f"/api/sources/{quote(str(obj_id), safe='')}/comments", dict(text=text, group_ids=group_ids or C.SKYPORTAL_GROUP_IDS or None))

    # own origin: SkyPortal keeps one annotation per origin, and "FLARE" belongs to the FLARE analysis service
    def annotation(self, obj_id, data: dict, origin="FLARE-agent", group_ids=None):
        return self._req("POST", f"/api/sources/{quote(str(obj_id), safe='')}/annotations", dict(origin=origin, data=data, group_ids=group_ids or C.SKYPORTAL_GROUP_IDS or None))


def publish(report, mode: str | None = None, client=None):
    """Post according to the mode: dry-run logs only; comment posts a comment and an annotation."""
    mode = C.MODE if mode is None else mode
    if mode == "dry-run":
        return dict(posted=False, mode=mode, would_post=report.comment())
    if mode in ("comment", "analysis"):
        sp = client or SkyPortal()
        c = sp.comment(report.obj_id, report.comment())
        a = sp.annotation(report.obj_id, report.annotation(), origin=report.ORIGIN)
        return dict(posted=True, mode=mode, comment=c.get("status"), annotation=a.get("status"))
    raise ValueError(f"unknown FLARE_MODE {mode}")
