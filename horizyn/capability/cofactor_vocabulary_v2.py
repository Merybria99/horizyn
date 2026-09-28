"""Versioned chemical cofactor groups for explicit annotation names.

The first ten positions preserve the v1 group order. Additional groups describe
chemical identities only: they do not imply a protein fold, binding architecture,
experimental validation, or cofactor absence. Metal oxidation states are grouped
by element; callers must retain the original names/evidence to preserve detail.

This matcher consumes cofactor-name fields, never arbitrary protein descriptions
or reaction text. It returns sparse known positives. Exporters, not this module,
assign the final ``unknown`` slot when a row has no supported cofactor evidence.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache


COFACTOR_VOCABULARY_VERSION = "circe_cofactor_v2"
UNKNOWN_LABEL = "unknown"
KNOWN_COFACTOR_LABELS: tuple[str, ...] = (
    "NAD_NADP",
    "FAD_FMN",
    "PLP",
    "TPP",
    "CoA",
    "SAM",
    "FeS",
    "heme",
    "quinone",
    "thiol_lipoate",
    "metal_Mg",
    "metal_Mn",
    "metal_Zn",
    "metal_Fe",
    "metal_Cu",
    "metal_Co",
    "metal_Ni",
    "metal_Ca",
    "metal_K",
    "metal_Na",
    "metal_generic",
    "metal_divalent",
    "cobalamin",
    "biotin",
    "molybdopterin_tungsten",
    "F420",
    "F430",
    "folate",
    "pyruvoyl",
    "dipyrromethane",
    "phosphopantetheine",
)
COFACTOR_LABELS = KNOWN_COFACTOR_LABELS + (UNKNOWN_LABEL,)
UNKNOWN_INDEX = len(KNOWN_COFACTOR_LABELS)


# Names, not the interpretation-heavy architecture map in cofactors.py, define
# this contract. Hyphens and underscores are normalized to spaces for matching.
_RULES = {
    "NAD_NADP": (
        r"\bNAD(?:P|\(P\))?H?\b|"
        r"\bnicotinamide(?:\s+adenine)?\s+dinucleotide(?:\s+phosphate)?\b"
    ),
    "FAD_FMN": (
        r"\b(?:(?:FAD|FMN)(?:H2?)?|flavin|riboflavin)\b|"
        r"\bflavin\s+(?:adenine\s+dinucleotide|mononucleotide)\b"
    ),
    "PLP": r"\b(?:PLP|pyridoxal|pyridoxamine)\b",
    "TPP": r"\b(?:TPP|ThDP)\b|\bthiamine?\s+(?:pyrophosphate|diphosphate)\b",
    "CoA": r"\bCoA\b|\bcoenzyme\s+A\b",
    "SAM": r"\b(?:SAM|AdoMet)\b|\bS\s+adenosyl\s*(?:L\s+)?methionine\b",
    "FeS": (
        r"\bFe\s*S\b|\biron\s+(?:sulfur|sulphur)\b|"
        r"(?<![a-z0-9])\d+\s*Fe\s*\d+\s*S(?![a-z0-9])"
    ),
    "heme": r"\b(?:heme|haem|siroheme|protoheme|hemin|ferriheme|ferroheme)\b",
    "quinone": r"\bPQQ\b|\b[a-z]*quin(?:one|ol)\b",
    "thiol_lipoate": (
        r"\b(?:glutathione|glutathionate|thioredoxin|glutaredoxin|"
        r"(?:dihydro)?lipoate|(?:dihydro)?lipoamide|mycothiol|bacillithiol|"
        r"GSH|GSSG|CoM)\b|\blipoic\s+acid\b|"
        # cob(II)alamin starts with a regex word boundary after 'cob'. Require
        # CoB to be an entire name, so cobalt oxidation notation cannot match.
        r"^CoB$|\bcoenzyme\s+[BM]\b(?!\s*12\b)|"
        r"\b(?:2\s+)?mercaptoethanesulfonate\b|"
        r"\b7\s+mercaptoheptanoylthreonine\s+phosphate\b"
    ),
    "cobalamin": (
        r"\b[a-z]*cob(?:\s*\(\s*(?:I|II|III)\s*\))?alamin\b|"
        r"\b[a-z]*cob(?:\s*\(\s*(?:I|II|III)\s*\))?amide\b|"
        r"\b(?:cobamamide|cobinamide|corrinoid)\b|"
        r"\b(?:vitamin|coenzyme)\s+B\s*12\b|\bB12\b"
    ),
    "biotin": r"\b(?:biotin|biotinate|carboxybiotin)\b",
    "molybdopterin_tungsten": (
        r"\b(?:molybdopterin|tungstopterin)\b|"
        r"\b(?:molybdenum|tungsten)\s+cofactor\b"
    ),
    "F420": r"\bF\s*420(?:H2)?\b|\b8\s+hydroxy\s+5\s+deazaflavin\b",
    "F430": r"\bF\s*430\b",
    "folate": r"\b(?:[a-z]*folate|[a-z]*folic\s+acid|THF|DHF)\b",
    # Free pyruvate is also a common reaction substrate. Its name alone does
    # not establish a protein-derived pyruvoyl group in this shared mapper.
    "pyruvoyl": r"\bpyruvoyl\b",
    "dipyrromethane": r"\bdipyrromethane\b",
    "phosphopantetheine": (
        r"\b(?:4\s*['′]?\s*)?phosphopantetheine\b|"
        r"\bpantetheine\s+4\s*['′]?\s*phosphate\b"
    ),
}
_COMPILED = {label: re.compile(pattern, re.IGNORECASE) for label, pattern in _RULES.items()}

_METAL_NAMES = {
    "Mg": "magnesium",
    "Mn": "manganese",
    "Zn": "zinc",
    "Fe": "iron|ferrous|ferric",
    "Cu": "copper|cuprous|cupric",
    "Co": "cobalt",
    "Ni": "nickel",
    "Ca": "calcium",
    "K": "potassium",
    "Na": "sodium",
}
_CHARGE = r"(?:\s*\(?\s*(?:\d*\s*\+|\+\s*\d+|(?i:[ivx]+))\s*\)?)?"
_METAL_COMPILED = {
    # Exact names avoid classifying iron-sulfur clusters, iron-containing heme,
    # and organometallic cofactors as independent simple metal-ion annotations.
    # Symbols remain case-sensitive: 'CO' is not the element symbol 'Co'.
    f"metal_{symbol}": re.compile(
        rf"(?i:(?:an?\s+)?)(?:{symbol}|(?i:{names})){_CHARGE}"
        r"(?i:(?:\s+(?:metal\s+)?(?:cation|ion))?)"
    )
    for symbol, names in _METAL_NAMES.items()
}
_GENERIC_METAL = re.compile(r"(?:an?\s+)?metal(?:\s+(?:cation|ion))?", re.IGNORECASE)
_DIVALENT_METAL = re.compile(
    r"(?:an?\s+)?divalent\s+(?:metal(?:\s+(?:cation|ion))?|cation)", re.IGNORECASE
)


@lru_cache(maxsize=8192)
def _name_groups(value: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r"[-_‐‑‒–—−]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    groups = {label for label, pattern in _COMPILED.items() if pattern.search(normalized)}
    groups.update(label for label, pattern in _METAL_COMPILED.items() if pattern.fullmatch(normalized))
    if _GENERIC_METAL.fullmatch(normalized):
        groups.add("metal_generic")
    if _DIVALENT_METAL.fullmatch(normalized):
        groups.add("metal_divalent")
    return frozenset(groups)


def cofactor_name_groups(values: Iterable[str]) -> set[str]:
    """Return known cofactor classes supported by explicit names, never unknown.

    Names are interpreted independently; no input or caller-owned evidence is
    changed. A string is accepted as one name for convenient direct use.
    """
    if isinstance(values, str):
        values = (values,)
    groups: set[str] = set()
    for value in values:
        if isinstance(value, str):
            groups.update(_name_groups(value))
    return groups


__all__ = [
    "COFACTOR_VOCABULARY_VERSION", "KNOWN_COFACTOR_LABELS", "COFACTOR_LABELS",
    "UNKNOWN_LABEL", "UNKNOWN_INDEX", "cofactor_name_groups",
]
