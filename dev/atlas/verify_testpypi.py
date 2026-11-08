"""Download the exact TestPyPI release and match it to the built artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def main() -> None:
    version, expected_directory, download_directory = sys.argv[1:]
    expected = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(expected_directory).iterdir()
        if path.name.endswith((".whl", ".tar.gz"))
    }
    if not expected:
        raise ValueError("No expected release artifacts")
    url = f"https://test.pypi.org/pypi/brian2-atlas/{version}/json"
    # Index propagation can lag the successful upload by a few seconds.
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                release = json.load(response)
            actual = {item["filename"]: item for item in release["urls"]}
            if set(actual) == set(expected):
                break
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
        if attempt == 11:
            raise RuntimeError("TestPyPI did not expose the exact expected artifact set")
        time.sleep(10)
    destination = Path(download_directory)
    destination.mkdir(parents=True, exist_ok=True)
    supported = set(sys_tags())
    matching_wheels = 0
    for name, digest in expected.items():
        item = actual[name]
        if item["digests"]["sha256"] != digest:
            raise ValueError(f"Index hash mismatch: {name}")
        with urllib.request.urlopen(item["url"], timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Downloaded bytes differ: {name}")
        # Verify every artifact, but only install the wheel for this interpreter.
        if name.endswith(".whl") and supported.intersection(parse_wheel_filename(name)[3]):
            (destination / name).write_bytes(content)
            matching_wheels += 1
        print(f"Verified {name}: {digest}")
    if matching_wheels != 1:
        raise ValueError(f"Expected one compatible wheel, found {matching_wheels}")


if __name__ == "__main__":
    main()
