#!/usr/bin/env python3
"""Transform query reactions with a fitted reaction-set chemistry schema."""

from __future__ import annotations

import argparse
import json

from horizyn.capability.reaction_set_features import materialize_reaction_set_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactions", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--cofactor-dictionary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = materialize_reaction_set_features(
        reactions_path=args.reactions,
        schema_path=args.schema,
        cofactor_dictionary_path=args.cofactor_dictionary,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
