"""Where things live.

Every path written in a config file is relative to the root of this release, so
a config keeps working wherever the package is unpacked. An absolute path is
left alone, which matters because the stores are large and people often keep
them on a different volume from the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
RUNS = ROOT / "runs"


def resolve(path, base=None):
    """Turn a config path into a real one."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return Path(base or ROOT) / p


def prepare(path):
    """Make sure a file's directory exists, then hand the path back."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
