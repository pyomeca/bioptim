"""
Download the Bioptim logo used by the series layer (``series_style.py``).

Source (pyomeca/biorbd_design): https://github.com/pyomeca/biorbd_design/blob/main/logo_png/bioptim_full.png
The PNG is git-ignored (global ``*.png`` rule, no binary commits): never commit it, just run this script.

    python docs/animations/assets/fetch_logo.py [--force]

Exit status is 0 when ``bioptim_logo.png`` is present and valid after the call, 1 otherwise.
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

URLS = [
    "https://raw.githubusercontent.com/pyomeca/biorbd_design/main/logo_png/bioptim_full.png",
    "https://github.com/pyomeca/biorbd_design/raw/main/logo_png/bioptim_full.png",
]
TARGET = Path(__file__).parent / "bioptim_logo.png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def is_valid_png(data: bytes) -> bool:
    return len(data) > 1000 and data.startswith(PNG_SIGNATURE)


def fetch_logo(force: bool = False, retries: int = 3, timeout: float = 30.0, verbose: bool = True) -> Path | None:
    """Download the logo if absent (or ``force``). Returns the path, or None if it could not be obtained."""
    if TARGET.exists() and not force and is_valid_png(TARGET.read_bytes()):
        return TARGET
    for attempt in range(retries):
        for url in URLS:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "bioptim-docs-series-layer"})
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    data = response.read()
                if not is_valid_png(data):
                    raise ValueError(f"not a valid PNG ({len(data)} bytes)")
                tmp = TARGET.with_suffix(".png.part")
                tmp.write_bytes(data)
                tmp.replace(TARGET)  # atomic: never leave a truncated file
                if verbose:
                    print(f"logo downloaded from {url} ({len(data)} bytes) -> {TARGET}")
                return TARGET
            except Exception as exc:  # network, HTTP, invalid content...
                if verbose:
                    print(f"[fetch_logo] {url}: {exc}", file=sys.stderr)
        time.sleep(1.5 * (attempt + 1))
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args()
    path = fetch_logo(force=args.force)
    if path is None:
        print("could not obtain the logo; the series layer will render without it", file=sys.stderr)
        sys.exit(1)
    print(path)
