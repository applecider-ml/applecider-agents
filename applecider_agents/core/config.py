"""Configuration by environment variables (see .env.example). No path in this file is
specific to a machine: the benchmark tables come from the public flare repository (cached
locally on first use), light curves from the broker archive through `flare.fetch`, and an
optional local data directory can override both."""
import os
from pathlib import Path

HOME = Path(os.environ.get("APPLECIDER_AGENTS_HOME", Path.home() / ".applecider_agents"))
CACHE = HOME / "cache"
LOG_DIR = Path(os.environ.get("APPLECIDER_AGENTS_LOG", HOME / "logs"))

# The FLARE benchmark tables (catalogue, splits, host / photo-z / Gaia-WISE / PS1 context) as
# published in the flare repository; pinned to a commit so results are reproducible.
FLARE_BENCHMARK_URL = os.environ.get("FLARE_BENCHMARK_URL", "https://raw.githubusercontent.com/applecider-ml/flare/c4b6377/benchmark")
# A local copy (e.g. a checkout's benchmark/ directory) takes precedence when set.
DATA_DIR = Path(os.environ["APPLECIDER_DATA_DIR"]) if os.environ.get("APPLECIDER_DATA_DIR") else None
# Optional directories of per-object event files <ZTFID>.npz, colon-separated; otherwise photometry is fetched.
EVENT_DIRS = [Path(p) for p in os.environ.get("APPLECIDER_EVENTS_DIRS", "").split(":") if p]
# Optional CSV of energies for classified objects (columns peakmag, energy) for the magnitude-matched threshold.
TEST_SCORES = Path(os.environ["APPLECIDER_TEST_SCORES"]) if os.environ.get("APPLECIDER_TEST_SCORES") else None

MODE = os.environ.get("FLARE_MODE", "dry-run")            # dry-run | comment
MODEL = os.environ.get("APPLECIDER_AGENTS_MODEL", "claude-opus-5")
EFFORT = os.environ.get("APPLECIDER_AGENTS_EFFORT", "high")
MAX_ITERATIONS = int(os.environ.get("APPLECIDER_AGENTS_MAX_ITER", "12"))
FALLBACKS = os.environ.get("APPLECIDER_AGENTS_FALLBACKS", "1") == "1"
BASE_RATE = float(os.environ.get("FLARE_BASE_RATE", "0.0326"))  # the benchmark's prior; a raw stream is far poorer in rarities
DAILY_CAP = int(os.environ.get("FLARE_DAILY_CAP", "200"))

SKYPORTAL_URL = os.environ.get("SKYPORTAL_URL", "")
SKYPORTAL_TOKEN = os.environ.get("SKYPORTAL_TOKEN", "")
SKYPORTAL_GROUP_IDS = [int(g) for g in os.environ.get("SKYPORTAL_GROUP_IDS", "").split(",") if g.strip()]
TNS_BOT_ID = os.environ.get("TNS_BOT_ID", ""); TNS_BOT_NAME = os.environ.get("TNS_BOT_NAME", ""); TNS_API_KEY = os.environ.get("TNS_API_KEY", "")
