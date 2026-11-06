"""Typed descriptors for Rust-side procedural topology initialization."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ClippedNormal:
    """A deterministic normal draw rejected outside optional inclusive bounds.

    Values retain Brian2 units until AtlasIR export, where they are checked against
    the destination variable (or delay) and converted to SI float64 values.
    """

    mean: object
    std: object
    minimum: object | None = None
    maximum: object | None = None


@dataclass(frozen=True)
class Uniform:
    """A deterministic continuous uniform draw over inclusive SI bounds."""

    minimum: object
    maximum: object
