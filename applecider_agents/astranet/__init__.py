"""ASTRANet model agent: spectroscopic classification and object-level novelty."""
from ..core.registry import ModelAgent
from .tools import TOOLS, astranet_analyze_spectrum
from .triage import SYSTEM, READING_GUIDE, SpectralReport, mock_triage

AGENT = ModelAgent(name="astranet", description="spectroscopic classifier (Ia, Ibc, SLSN, II, AGN, CV, TDE; twelve members, redshift optional) with an object-level novelty p-value and alarms at 1, 5, 10 and 15 % on every spectrum of the source.",
                   tools=TOOLS, triage=mock_triage, system_prompt=SYSTEM, reading_guide=READING_GUIDE, report_cls=SpectralReport, mcp_extra=[astranet_analyze_spectrum]).register()
