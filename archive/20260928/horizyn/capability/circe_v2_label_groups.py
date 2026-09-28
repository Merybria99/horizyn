"""Conservative cofactor-name matching for new CIRCE-v2 annotation exports.

Unlike substring matching, these aliases do not turn vanadium into NAD or
coenzyme B12 into coenzyme B. Unrecognized names remain unknown.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


_RULES = {
    "NAD_NADP": r"\b(?:NADP?H?|nicotinamide(?:\s+adenine)?\s+dinucleotide)\b",
    "FAD_FMN": r"\b(?:FAD|FMN|flavin|riboflavin)\b",
    "PLP": r"\b(?:PLP|pyridoxal|pyridoxamine)\b",
    "TPP": r"\bTPP\b|\bthiamine?\s+(?:pyrophosphate|diphosphate)\b",
    "CoA": r"\bCoA\b|\bcoenzyme\s+A\b",
    "SAM": r"\bSAM\b|\bS\s+adenosyl\s+(?:L\s+)?methionine\b",
    "FeS": r"\bFeS\b|\biron\s+(?:sulfur|sulphur)\b|(?<![a-z0-9])[234]\s*Fe\s*[234]\s*S(?![a-z0-9])",
    "heme": r"\b(?:heme|haem|siroheme|protoheme|hemin)\b",
    "quinone": r"\bPQQ\b|\b[a-z]*quin(?:one|ol)\b",
    "thiol_lipoate": r"\b(?:glutathione|thioredoxin|glutaredoxin|lipoate|lipoamide|mycothiol|bacillithiol|CoB|CoM)\b|\blipoic\s+acid\b|\bcoenzyme\s+[BM]\b",
}
_COMPILED = {label: re.compile(pattern, re.IGNORECASE) for label, pattern in _RULES.items()}


def cofactor_name_groups(values: Iterable[str]) -> set[str]:
    normalized = [re.sub(r"[-_‐‑–]+", " ", str(value)) for value in values]
    return {
        label for label, pattern in _COMPILED.items()
        if any(pattern.search(value) for value in normalized)
    }
