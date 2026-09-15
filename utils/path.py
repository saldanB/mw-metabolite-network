from pathlib import Path

def find_root(marker="pyproject.toml", start=None):
    start = Path(start or Path.cwd()).resolve()
    for p in (start, *start.parents):
        if (p / marker).exists():
            return p
    raise RuntimeError(f"{marker} not found above {start}")