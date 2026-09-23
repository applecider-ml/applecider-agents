"""FLARE model agent: photometric classification, conformal sets, anomaly energy."""
from ..core.registry import ModelAgent
from .tools import TOOLS, flare_analyze_photometry
from .triage import SYSTEM, READING_GUIDE, mock_triage

AGENT = ModelAgent(name="flare", description="photometric classifier (SN Ia, SN CC, SLSN, AGN, TDE, CV) with 90% conformal sets and a blind anomaly energy on ZTF light curves plus alert-time catalogue context.",
                   tools=TOOLS, triage=mock_triage, system_prompt=SYSTEM, reading_guide=READING_GUIDE, mcp_extra=[flare_analyze_photometry]).register()
