"""Adapter for ingesting BOOM broker alerts into AppleCiDEr agent workflows.

Converts broker alert packets (detections + pre-staged cross-matches) into
the standardized event array and context dictionary, seeding the agent core.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .client import BoomConnectorClient
from ..core import data as DATA
from flare.fetch import to_events


def adapt_boom_object(
    obj_id: str,
    client: Optional[BoomConnectorClient] = None,
    rows: Optional[List[Tuple[float, int, float, float]]] = None,
    ctx: Optional[Dict[str, Any]] = None,
    seed_core: bool = True
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Adapt a broker alert candidate into AppleCiDEr event array and context.

    Args:
        obj_id: Object identifier (e.g. 'ZTF25abhoijc')
        client: Optional BoomConnectorClient instance
        rows: Optional pre-fetched photometry rows [(mjd, fid, mag, magerr), ...]
        ctx: Optional pre-staged crossmatches dict (PS1, Gaia, WISE, photo-z)
        seed_core: If True, seeds applecider_agents.core.data with this object

    Returns:
        (events_array, context_dict)
    """
    client = client or BoomConnectorClient()

    # 1. Obtain photometry rows
    if rows is None:
        rows = client.get_photometry_rows(obj_id)

    # 2. Convert rows into AppleCiDEr (N, 15) event array
    ev = to_events(rows) if rows else None

    # 3. Obtain pre-staged catalogue context
    if ctx is None:
        ctx = client.get_crossmatches(obj_id)

    # Ensure context has source attribution
    clean_ctx = {k: v for k, v in ctx.items() if v is not None}
    clean_ctx["_source"] = "boom_connector"

    # 4. Seed the core cache for zero-latency agent execution
    if seed_core and ev is not None:
        DATA.seed(obj_id, ev, clean_ctx)

    return ev, clean_ctx
