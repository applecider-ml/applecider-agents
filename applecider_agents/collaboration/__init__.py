"""Multi-Agent collaborative investigation and transient discovery for AppleCiDEr."""
from .schemas import SpecialistAssessment, DebateMessage, DiscoveryReport
from .specialists import TriageSpecialist, PhotometrySpecialist, AnomalySpecialist, ContextSpecialist
from .lead_scientist import LeadScientist
from .workflow import collaborate

__all__ = [
    "SpecialistAssessment",
    "DebateMessage",
    "DiscoveryReport",
    "TriageSpecialist",
    "PhotometrySpecialist",
    "AnomalySpecialist",
    "ContextSpecialist",
    "LeadScientist",
    "collaborate"
]
