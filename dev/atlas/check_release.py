"""Gate production release tags; ordinary PR/manual builds remain dry runs."""
import os
from pathlib import Path
import tomllib

from packaging.version import Version


def validate_release(version: str, ref: str) -> None:
    parsed = Version(version)
    if not ref.startswith("refs/tags/"):
        return
    if ref != f"refs/tags/v{version}":
        raise ValueError("Release tag must exactly match the package version")
    if parsed.is_prerelease or parsed.is_devrelease or parsed.local is not None:
        raise ValueError("Production releases require a stable public version")


def main() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    version = project["version"]
    ref = os.environ["GITHUB_REF"]
    validate_release(version, ref)
    if ref.startswith("refs/tags/"):
        expected_docs = f"https://github.com/RockLi/brian2-atlas/blob/v{version}/documentation/README.md"
        if project["urls"]["Documentation"] != expected_docs:
            raise ValueError("Documentation URL must identify the production release tag")
        if "/blob/testpypi-v" in Path("README.pypi.md").read_text():
            raise ValueError("Update the package description's TestPyPI links for production")
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"version={version}\n")


if __name__ == "__main__":
    main()
