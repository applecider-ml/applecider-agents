"""Client for the BOOM alert broker API.

Handles authentication, alert querying from ZTF and Rubin collections,
and extraction of pre-staged cross-matches.
"""
from __future__ import annotations
import os
import json
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Any, Tuple


def _http_json(url: str, data: Optional[dict] = None, headers: Optional[dict] = None, form: bool = False) -> dict:
    h = {"User-Agent": "applecider-agents/boom-connector"}
    if headers:
        h.update(headers)
    body = None
    if data is not None:
        if form:
            h["Content-Type"] = "application/x-www-form-urlencoded"
            body = urllib.parse.urlencode(data).encode("utf-8")
        else:
            h["Content-Type"] = "application/json"
            body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=h)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


class BoomConnectorClient:
    """Connector client for the BOOM alert broker API."""

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None,
                 username: Optional[str] = None, password: Optional[str] = None,
                 fixture_file: Optional[str] = None):
        self.url = (url or os.environ.get("BOOM_URL", "")).rstrip("/")
        self.token = token or os.environ.get("BOOM_TOKEN")
        self.username = username or os.environ.get("BOOM_USERNAME")
        self.password = password or os.environ.get("BOOM_PASSWORD")
        self.fixture_file = fixture_file
        self._fixtures: Optional[Dict[str, Any]] = None

        if self.fixture_file and os.path.exists(self.fixture_file):
            with open(self.fixture_file, "r") as f:
                self._fixtures = json.load(f)

    @property
    def is_configured(self) -> bool:
        """True if either a live BOOM endpoint or local fixture data is available."""
        return bool(self.url or self._fixtures)

    def _get_token(self) -> str:
        if self.token:
            return self.token
        if not (self.username and self.password):
            raise RuntimeError("BOOM authentication requires BOOM_TOKEN or BOOM_USERNAME/BOOM_PASSWORD.")
        resp = _http_json(f"{self.url}/auth",
                          data={"username": self.username, "password": self.password},
                          form=True)
        self.token = resp.get("access_token", "")
        return self.token

    def _auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}"}

    def find(self, catalog: str, filt: dict, projection: Optional[dict] = None, limit: int = 100) -> List[dict]:
        """Execute a query against a BOOM broker catalog (e.g. ZTF_alerts, ZTF_alerts_aux)."""
        if not self.url:
            if self._fixtures:
                # Return matching fixture objects
                oid = filt.get("objectId") or filt.get("_id")
                if oid and oid in self._fixtures:
                    return [self._fixtures[oid]]
                return list(self._fixtures.values())[:limit]
            raise RuntimeError("BOOM_URL is not configured and no local fixture is provided.")

        payload = {"catalog_name": catalog, "filter": filt, "limit": limit}
        if projection:
            payload["projection"] = projection
        resp = _http_json(f"{self.url}/queries/find", data=payload, headers=self._auth_headers())
        return resp.get("data", resp) or []

    def get_photometry_rows(self, obj_id: str) -> List[Tuple[float, int, float, float]]:
        """Fetch deduplicated photometry (mjd, fid, mag, magerr) for an object from the broker."""
        if self._fixtures and obj_id in self._fixtures:
            data = self._fixtures[obj_id]
            mjd0 = data.get("mjd0", 59000.0)
            rows = []
            for b_name, fid in (("g", 1), ("r", 2), ("i", 3)):
                for pt in data.get("mag", {}).get(b_name, []):
                    # pt is [dt, mag, magerr]
                    dt, mag, magerr = pt[0], pt[1], pt[2]
                    rows.append((mjd0 + dt, fid, mag, magerr))
            return sorted(rows, key=lambda x: x[0])

        dets = {}
        # 1. Primary alert candidates
        for doc in self.find("ZTF_alerts", {"objectId": obj_id},
                             {"candidate.jd": 1, "candidate.fid": 1,
                              "candidate.magpsf": 1, "candidate.sigmapsf": 1}):
            c = doc.get("candidate", {})
            if c.get("magpsf") is not None and c.get("fid") in (1, 2, 3):
                dets[round(c["jd"], 5)] = (c["jd"] - 2400000.5, c["fid"], c["magpsf"], c["sigmapsf"])

        # 2. Previous candidate history
        for doc in self.find("ZTF_alerts_aux", {"_id": obj_id}, {"prv_candidates": 1}):
            for c in doc.get("prv_candidates", []) or []:
                if c.get("magpsf") is not None and c.get("fid") in (1, 2, 3):
                    dets.setdefault(round(c["jd"], 5), (c["jd"] - 2400000.5, c["fid"], c["magpsf"], c["sigmapsf"]))

        return sorted(dets.values(), key=lambda x: x[0])

    def get_crossmatches(self, obj_id: str) -> Dict[str, Any]:
        """Extract pre-staged catalogue cross-matches attached to the object in the broker."""
        if self._fixtures and obj_id in self._fixtures:
            return self._fixtures[obj_id].get("ctx", {})

        # Query broker's pre-computed crossmatches table or aux document
        docs = self.find("ZTF_alerts_aux", {"_id": obj_id}, {"crossmatches": 1, "ctx": 1})
        if docs:
            doc = docs[0]
            return doc.get("crossmatches") or doc.get("ctx") or {}
        return {}

    def query_recent_candidates(self, min_detections: int = 2, max_mag: float = 20.0,
                                limit: int = 50) -> List[str]:
        """Retrieve recent unclassified candidate object IDs meeting screening criteria."""
        if self._fixtures:
            return [oid for oid, d in self._fixtures.items() if d.get("n_det", 0) >= min_detections][:limit]

        query = {
            "candidate.magpsf": {"$lte": max_mag},
            "candidate.drb": {"$gte": 0.4}
        }
        docs = self.find("ZTF_alerts", query, {"objectId": 1}, limit=limit * 2)
        oids = []
        for d in docs:
            oid = d.get("objectId")
            if oid and oid not in oids:
                oids.append(oid)
            if len(oids) >= limit:
                break
        return oids
