"""The MSight agent. Importing it loads .env first (so every module sees its
settings at import time) and makes the auto-labeling engine's modules importable."""
from dotenv import load_dotenv

load_dotenv()

from msight_agent.shared import paths as _paths  # noqa: E402,F401
