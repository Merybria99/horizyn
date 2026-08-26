"""Strict v2 provenance contracts for Horizyn-generated artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping


ARTIFACT_SCHEMA_VERSION = 2
VALID_DIRECTIONS = {"reaction_to_enzyme", "enzyme_to_reaction", "both", "not_applicable"}


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Artifact input is not a readable file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    """Hash an ordered string sequence without delimiter ambiguity."""

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def fingerprint_file(path: str | Path) -> dict[str, Any]:
    """Return strict content identity for one artifact input."""

    source = Path(path).expanduser().resolve()
    stat_result = source.stat()
    if not source.is_file():
        raise ValueError(f"Artifact input must be a regular file: {source}")
    return {
        "path": str(source),
        "size": int(stat_result.st_size),
        "sha256": sha256_file(source),
    }


def current_git_revision(project_root: str | Path | None = None) -> str:
    """Resolve the exact Git revision used to generate an artifact."""

    root = Path(project_root or Path(__file__).resolve().parents[1])
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


@dataclass(frozen=True)
class ArtifactManifestV2:
    """Machine-validatable provenance attached to every v2 artifact."""

    artifact_type: str
    role: str
    direction: str
    inputs: Mapping[str, Any]
    code_revision: str
    config_sha256: str
    candidate_ids_sha256: str | None = None
    score_type: str | None = None
    higher_is_better: bool | None = None
    row_count: int | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    seed: int | None = None
    schema_version: int = field(default=ARTIFACT_SCHEMA_VERSION, init=False)

    def validate(self) -> None:
        if not self.artifact_type.strip():
            raise ValueError("artifact_type must be non-empty")
        if not self.role.strip():
            raise ValueError("role must be non-empty")
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(VALID_DIRECTIONS)}, got {self.direction!r}"
            )
        if not self.code_revision.strip():
            raise ValueError("code_revision must be non-empty")
        _require_sha256(self.config_sha256, "config_sha256")
        if self.candidate_ids_sha256 is not None:
            _require_sha256(self.candidate_ids_sha256, "candidate_ids_sha256")
        if self.score_type is not None and self.higher_is_better is None:
            raise ValueError("score artifacts must declare higher_is_better")
        if self.row_count is not None and self.row_count < 0:
            raise ValueError("row_count must be non-negative")
        if self.shape is not None and any(value < 0 for value in self.shape):
            raise ValueError("shape dimensions must be non-negative")
        if not isinstance(self.inputs, Mapping):
            raise TypeError("inputs must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        if self.shape is not None:
            payload["shape"] = list(self.shape)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactManifestV2":
        if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Expected artifact schema v{ARTIFACT_SCHEMA_VERSION}; "
                f"got {payload.get('schema_version')!r}"
            )
        values = dict(payload)
        values.pop("schema_version", None)
        if values.get("shape") is not None:
            values["shape"] = tuple(int(value) for value in values["shape"])
        manifest = cls(**values)
        manifest.validate()
        return manifest


@dataclass(frozen=True)
class SplitRoleManifestV2:
    """Identity and role contract for one benchmark split."""

    split: str
    pairs_sha256: str
    candidate_ids_sha256: str
    positive_pair_count: int
    query_count: int
    candidate_count: int
    schema_version: int = field(default=ARTIFACT_SCHEMA_VERSION, init=False)

    def validate(self) -> None:
        if not self.split.strip():
            raise ValueError("split must be non-empty")
        for name, value in (
            ("pairs_sha256", self.pairs_sha256),
            ("candidate_ids_sha256", self.candidate_ids_sha256),
        ):
            _require_sha256(value, name)
        if self.positive_pair_count <= 0:
            raise ValueError("positive_pair_count must be positive")
        if self.query_count <= 0:
            raise ValueError("query_count must be positive")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def write_manifest(path: str | Path, manifest: ArtifactManifestV2) -> None:
    """Atomically write one validated artifact manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
