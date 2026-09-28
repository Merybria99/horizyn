"""Conservative cofactor extraction rules.

ChEBI:23357 is a role term. The concrete cofactors are incoming
``has role cofactor`` assertions, so the static list below is a local fallback
snapshot of those ChEBI role fillers. The curated aliases preserve the broad
labels used by reaction-type rules; otherwise a ChEBI-listed cofactor falls
back to a stable exact label.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any
import unicodedata

NO_COFACTOR_LABEL = "no_cofactor"
NONE_OR_UNKNOWN_ARCHITECTURE_BIN = "none_or_unknown"
UNKNOWN_CHEMISTRY_BIN = "unknown"

COFACTOR_ARCHITECTURE_BINS: tuple[str, ...] = (
    NONE_OR_UNKNOWN_ARCHITECTURE_BIN,
    "rossmann_NAD_NADP",
    "flavin_FAD_FMN",
    "PLP_dependent",
    "TPP_dependent",
    "P_loop_NTP_kinase",
    "simple_metal_binding",
    "FeS_metal_cluster",
    "heme_tetrapyrrole",
    "CoA_acyl_carrier",
    "SAM_methyl_or_radical",
    "sugar_nucleotide_GT",
    "thiol_or_protein_redox",
    "quinone_PQQ_redox",
    "molybdopterin_redox",
    "biotin_carboxylation",
    "ascorbate_pterin_redox",
    "carotenoid_redox",
    "C1_folate_methanogen_carrier",
    "other_organic_group_transfer",
)

COFACTOR_CHEMISTRY_BINS: tuple[str, ...] = (
    "redox",
    "group_transfer",
    "phosphoryl_transfer",
    "radical",
    "metal_catalysis",
    "structural_metal",
    "conjugation",
    UNKNOWN_CHEMISTRY_BIN,
)

_COFACTOR_ARCHITECTURE_BIN_MAP: dict[str, tuple[str, ...]] = {
    NO_COFACTOR_LABEL: (NONE_OR_UNKNOWN_ARCHITECTURE_BIN,),
    "NAD": ("rossmann_NAD_NADP",),
    "NADP": ("rossmann_NAD_NADP",),
    "FAD": ("flavin_FAD_FMN",),
    "FMN": ("flavin_FAD_FMN",),
    "flavin": ("flavin_FAD_FMN",),
    "PLP": ("PLP_dependent",),
    "TPP": ("TPP_dependent",),
    "ATP": ("P_loop_NTP_kinase",),
    "GTP": ("P_loop_NTP_kinase",),
    "CoA": ("CoA_acyl_carrier",),
    "SAM": ("SAM_methyl_or_radical",),
    "sugar_nucleotide": ("sugar_nucleotide_GT",),
    "FeS_cluster": ("FeS_metal_cluster",),
    "heme": ("heme_tetrapyrrole",),
    "siroheme": ("heme_tetrapyrrole",),
    "F430": ("heme_tetrapyrrole",),
    "cobalamin": ("heme_tetrapyrrole",),
    "cobamamide": ("heme_tetrapyrrole",),
    "methylcobalamin": ("heme_tetrapyrrole",),
    "glutathione": ("thiol_or_protein_redox",),
    "mycothiol": ("thiol_or_protein_redox",),
    "bacillithiol": ("thiol_or_protein_redox",),
    "lipoate": ("thiol_or_protein_redox",),
    "ferredoxin": ("thiol_or_protein_redox",),
    "quinone": ("quinone_PQQ_redox",),
    "PQQ": ("quinone_PQQ_redox",),
    "molybdopterin": ("molybdopterin_redox",),
    "THF": ("C1_folate_methanogen_carrier",),
    "biotin": ("biotin_carboxylation",),
    "ascorbate": ("ascorbate_pterin_redox",),
    "pterin": ("ascorbate_pterin_redox",),
    "carotenoid": ("carotenoid_redox",),
    "F420": ("other_organic_group_transfer",),
    "CoM": ("other_organic_group_transfer",),
    "CoB": ("other_organic_group_transfer",),
    "methanofuran": ("C1_folate_methanogen_carrier",),
    "methanopterin": ("C1_folate_methanogen_carrier",),
    "pantetheine_4_phosphate": ("CoA_acyl_carrier",),
    "pantetheine_4_phosphate_2": ("CoA_acyl_carrier",),
    "pyridoxal": ("PLP_dependent",),
    "pyridoxine": ("PLP_dependent",),
    "nicotinamide": ("rossmann_NAD_NADP",),
    "fe_coproporphyrin_iii_4": ("heme_tetrapyrrole",),
    "5_hydroxybenzimidazolylcob_i_amide_1": ("heme_tetrapyrrole",),
    "ni_ii_pyridinium_3_5_bisthiocarboxylate_mononucleotide": (
        "other_organic_group_transfer",
    ),
    "ni_ii_pyridinium_3_5_bisthiocarboxylate_mononucleotide_1": (
        "other_organic_group_transfer",
    ),
    "premycofactocin": ("other_organic_group_transfer",),
    "6_7_dimethyl_8_1_d_ribityl_lumazine_1": ("other_organic_group_transfer",),
    "myo_inositol_hexakisphosphate": ("other_organic_group_transfer",),
    "myo_inositol_hexakisphosphate_12": ("other_organic_group_transfer",),
}

_COFACTOR_CHEMISTRY_BIN_MAP: dict[str, tuple[str, ...]] = {
    NO_COFACTOR_LABEL: (UNKNOWN_CHEMISTRY_BIN,),
    "NAD": ("redox",),
    "NADP": ("redox",),
    "FAD": ("redox",),
    "FMN": ("redox",),
    "flavin": ("redox",),
    "PLP": ("group_transfer",),
    "TPP": ("group_transfer",),
    "ATP": ("phosphoryl_transfer", "group_transfer"),
    "GTP": ("phosphoryl_transfer", "group_transfer"),
    "CoA": ("group_transfer",),
    "SAM": ("group_transfer", "radical"),
    "sugar_nucleotide": ("group_transfer",),
    "FeS_cluster": ("redox", "radical"),
    "heme": ("redox", "metal_catalysis"),
    "siroheme": ("redox", "metal_catalysis"),
    "F430": ("redox", "metal_catalysis"),
    "cobalamin": ("group_transfer", "radical"),
    "cobamamide": ("group_transfer", "radical"),
    "methylcobalamin": ("group_transfer", "radical"),
    "glutathione": ("conjugation", "redox"),
    "mycothiol": ("conjugation", "redox"),
    "bacillithiol": ("conjugation", "redox"),
    "lipoate": ("redox", "group_transfer"),
    "ferredoxin": ("redox",),
    "quinone": ("redox",),
    "PQQ": ("redox",),
    "molybdopterin": ("redox", "group_transfer", "metal_catalysis"),
    "THF": ("group_transfer",),
    "biotin": ("group_transfer",),
    "ascorbate": ("redox",),
    "pterin": ("redox",),
    "carotenoid": ("redox",),
    "F420": ("redox",),
    "CoM": ("group_transfer",),
    "CoB": ("group_transfer",),
    "methanofuran": ("group_transfer",),
    "methanopterin": ("group_transfer",),
    "pantetheine_4_phosphate": ("group_transfer",),
    "pantetheine_4_phosphate_2": ("group_transfer",),
    "pyridoxal": ("group_transfer",),
    "pyridoxine": ("group_transfer",),
    "nicotinamide": ("redox",),
    "fe_coproporphyrin_iii_4": ("redox", "metal_catalysis"),
    "5_hydroxybenzimidazolylcob_i_amide_1": ("group_transfer", "radical"),
    "ni_ii_pyridinium_3_5_bisthiocarboxylate_mononucleotide": (
        "redox",
        "metal_catalysis",
    ),
    "ni_ii_pyridinium_3_5_bisthiocarboxylate_mononucleotide_1": (
        "redox",
        "metal_catalysis",
    ),
    "premycofactocin": ("redox",),
    "6_7_dimethyl_8_1_d_ribityl_lumazine_1": ("group_transfer",),
    "myo_inositol_hexakisphosphate": ("group_transfer",),
    "myo_inositol_hexakisphosphate_12": ("group_transfer",),
    "hydrogenphosphate": ("phosphoryl_transfer",),
    "hydrogen_peroxide": ("redox",),
    "hydrogencarbonate": ("group_transfer",),
    "pyruvate": ("group_transfer",),
    "pyruvic_acid": ("group_transfer",),
}

_METAL_CATALYTIC_LABELS: frozenset[str] = frozenset(
    {
        "Zn2+",
        "Fe2+",
        "Fe3+",
        "Mn2+",
        "Cu2+",
        "Ni2+",
        "Co2+",
        "Mo",
        "W",
    }
)

CURATED_COFACTOR_ALIASES: dict[str, list[str]] = {
    "NAD": ["NAD", "NAD+", "NADH", "nicotinamide adenine dinucleotide"],
    "NADP": ["NADP", "NADP+", "NADPH"],
    "FAD": ["FAD", "FADH2", "flavin adenine dinucleotide"],
    "FMN": ["FMN", "FMNH2", "flavin mononucleotide"],
    "PLP": ["PLP", "pyridoxal phosphate", "pyridoxal 5'-phosphate"],
    "TPP": ["thiamine diphosphate", "thiamine pyrophosphate", "TPP", "ThDP"],
    "ATP": ["ATP", "ADP", "AMP"],
    "GTP": ["GTP", "GDP", "GMP"],
    "SAM": ["S-adenosyl-L-methionine", "S-adenosylmethionine", "SAM", "SAH"],
    "sugar_nucleotide": [
        "UDP-sugar",
        "UDP-glucose",
        "UDP-D-glucose",
        "UDP-alpha-D-glucose",
        "UDP-galactose",
        "UDP-N-acetylglucosamine",
        "UDP-N-acetylgalactosamine",
        "GDP-sugar",
        "GDP-mannose",
        "GDP-fucose",
        "CMP-N-acetylneuraminate",
        "dTDP-glucose",
    ],
    "glutathione": ["glutathione", "glutathionate", "GSH", "GSSG"],
    "bacillithiol": ["bacillithiol"],
    "mycothiol": ["mycothiol"],
    "CoA": ["coenzyme A", "CoA", "acetyl-CoA", "acyl-CoA"],
    "biotin": ["biotin", "carboxybiotin"],
    "heme": ["heme", "haem", "CHEBI:30413"],
    "ferredoxin": ["ferredoxin"],
    "quinone": ["quinone", "ubiquinone", "menaquinone"],
    "PQQ": ["pyrroloquinoline quinone", "pyrroloquinoline cofactor", "PQQ"],
    "molybdopterin": [
        "molybdopterin",
        "molybdopterin cofactor",
        "Mo-molybdopterin",
        "Mo-molybdopterin cytosine dinucleotide",
        "Mo-molybdopterin guanine dinucleotide",
        "Mo-bis(molybdopterin guanine dinucleotide)",
        "W-bis(molybdopterin guanine dinucleotide)",
        "CHEBI:60537",
        "CHEBI:60539",
        "CHEBI:71302",
        "CHEBI:71308",
        "CHEBI:71310",
    ],
    "flavin": ["flavin", "oxidized flavin", "CHEBI:60531"],
    "metal": [
        "metal",
        "metal cation",
        "iron cation",
        "divalent metal cation",
        "CHEBI:24875",
        "CHEBI:25213",
        "CHEBI:60240",
        "CHEBI:229784",
        "CHEBI:229785",
        "CHEBI:49591",
        "CHEBI:49890",
    ],
    "Mg2+": ["Mg2+", "Mg(2+)", "magnesium", "CHEBI:18420"],
    "Zn2+": ["Zn2+", "Zn(2+)", "zinc", "CHEBI:29105"],
    "Fe2+": ["Fe2+", "Fe(2+)", "iron(ii)", "ferrous", "CHEBI:29033"],
    "Fe3+": ["Fe3+", "Fe(3+)", "iron(iii)", "ferric", "CHEBI:29034"],
    "Mn2+": ["Mn2+", "Mn(2+)", "manganese"],
    "Cu2+": ["Cu2+", "Cu(2+)", "copper"],
    "Ni2+": ["Ni2+", "Ni(2+)", "nickel", "CHEBI:49786"],
    "Ca2+": ["Ca2+", "Ca(2+)", "calcium", "CHEBI:29108"],
    "Co2+": ["Co2+", "Co(2+)", "cobalt", "CHEBI:48828"],
    "Pr3+": ["Pr3+", "Pr(3+)", "praseodymium", "CHEBI:229784"],
    "Nd3+": ["Nd3+", "Nd(3+)", "neodymium", "CHEBI:229785"],
    "Eu3+": ["Eu3+", "Eu(3+)", "europium", "CHEBI:49591"],
    "Sm3+": ["Sm3+", "Sm(3+)", "samarium", "CHEBI:49890"],
    "K+": ["K+", "potassium", "CHEBI:29103"],
    "Na+": ["Na+", "sodium", "CHEBI:29101"],
    "FeS_cluster": [
        "iron-sulfur",
        "Fe-S",
        "FeS",
        "[2Fe-2S] cluster",
        "[3Fe-4S] cluster",
        "CHEBI:190135",
        "CHEBI:21137",
        "CHEBI:30408",
    ],
    "THF": ["tetrahydrofolate", "THF"],
    "lipoate": ["lipoate", "lipoic acid"],
    "cobalamin": ["cobalamin", "vitamin B12"],
    "ascorbate": ["ascorbate", "ascorbic acid"],
    "pterin": ["pterin", "tetrahydropteridine", "sapropterin"],
    "chlorophyll": ["chlorophyll"],
    "carotenoid": ["carotene", "zeaxanthin", "echinenone"],
    "F430": ["coenzyme F430", "F430"],
    "siroheme": ["siroheme", "sirohydrochlorin"],
    "F420": ["coenzyme F420", "F420", "8-hydroxy-5-deazaflavin"],
    "CoM": ["coenzyme M", "2-mercaptoethanesulfonate", "mercaptoethanesulfonate"],
    "CoB": ["coenzyme B", "7-mercaptoheptanoylthreonine phosphate"],
    "methanofuran": ["methanofuran"],
    "methanopterin": ["methanopterin", "tetrahydromethanopterin"],
}

# Static fallback from the incoming ``has role cofactor`` table on
# https://www.ebi.ac.uk/chebi/CHEBI:23357, parsed 2026-06-30.
CHEBI_COFACTOR_ROLE_TERMS: tuple[tuple[str, str], ...] = (
    ("CHEBI:83088", "(R)-lipoate"),
    ("CHEBI:30314", "(R)-lipoic acid"),
    ("CHEBI:15636", "(6R)-5,10-methylenetetrahydrofolate(2-)"),
    ("CHEBI:1989", "(6R)-5,10-methylenetetrahydrofolic acid"),
    ("CHEBI:57453", "(6S)-5,6,7,8-tetrahydrofolate(2-)"),
    ("CHEBI:15635", "(6S)-5,6,7,8-tetrahydrofolic acid"),
    ("CHEBI:16680", "S-adenosyl-L-homocysteine"),
    ("CHEBI:15414", "S-adenosyl-L-methionine"),
    ("CHEBI:59789", "S-adenosyl-L-methionine zwitterion"),
    ("CHEBI:17401", "myo-inositol hexakisphosphate"),
    ("CHEBI:58130", "myo-inositol hexakisphosphate(12-)"),
    ("CHEBI:38290", "L-ascorbate"),
    ("CHEBI:29073", "L-ascorbic acid"),
    ("CHEBI:16509", "1,4-benzoquinone"),
    ("CHEBI:80214", "3'-hydroxyechinenone"),
    ("CHEBI:17679", "5-hydroxybenzimidazolylcob(I)amide"),
    ("CHEBI:60494", "5-hydroxybenzimidazolylcob(I)amide(1-)"),
    ("CHEBI:28889", "5,6,7,8-tetrahydropteridine"),
    ("CHEBI:52020", "6-decylubiquinone"),
    ("CHEBI:40260", "6-hydroxy-FAD"),
    ("CHEBI:60470", "6-hydroxy-FAD(3-)"),
    ("CHEBI:17601", "6,7-dimethyl-8-(1-D-ribityl)lumazine"),
    ("CHEBI:58201", "6,7-dimethyl-8-(1-D-ribityl)lumazine(1-)"),
    ("CHEBI:16027", "adenosine 5'-monophosphate"),
    ("CHEBI:456215", "adenosine 5'-monophosphate(2-)"),
    ("CHEBI:28938", "ammonium"),
    ("CHEBI:15422", "ATP"),
    ("CHEBI:57299", "ATP(3-)"),
    ("CHEBI:30616", "ATP(4-)"),
    ("CHEBI:61338", "bacillithiol"),
    ("CHEBI:37136", "barium(2+)"),
    ("CHEBI:15956", "biotin"),
    ("CHEBI:57586", "biotinate"),
    ("CHEBI:30402", "bis(molybdopterin)tungsten cofactor"),
    ("CHEBI:48775", "cadmium(2+)"),
    ("CHEBI:29108", "calcium(2+)"),
    ("CHEBI:48782", "cerium(3+)"),
    ("CHEBI:17996", "chloride"),
    ("CHEBI:18230", "chlorophyll a"),
    ("CHEBI:58416", "chlorophyll a(1-)"),
    ("CHEBI:27888", "chlorophyll b"),
    ("CHEBI:61721", "chlorophyll b(1-)"),
    ("CHEBI:15982", "cob(I)alamin"),
    ("CHEBI:60488", "cob(I)alamin(1-)"),
    ("CHEBI:16304", "cob(II)alamin"),
    ("CHEBI:30411", "cobalamin"),
    ("CHEBI:48828", "cobalt(2+)"),
    ("CHEBI:49415", "cobalt(3+)"),
    ("CHEBI:18408", "cobamamide"),
    ("CHEBI:28265", "coenzyme F430"),
    ("CHEBI:60540", "coenzyme F430(4-)"),
    ("CHEBI:23378", "copper cation"),
    ("CHEBI:49552", "copper(1+)"),
    ("CHEBI:29036", "copper(2+)"),
    ("CHEBI:33221", "corrin"),
    ("CHEBI:33913", "corrinoid"),
    ("CHEBI:72953", "decylplastoquinone"),
    ("CHEBI:17803", "dehydro-D-arabinono-1,4-lactone"),
    ("CHEBI:35169", "dihydrogenvanadate"),
    ("CHEBI:17694", "dihydrolipoamide"),
    ("CHEBI:42121", "dipyrromethane cofactor"),
    ("CHEBI:60342", "dipyrromethane cofactor(4-)"),
    ("CHEBI:73113", "divinyl chlorophyll a"),
    ("CHEBI:73095", "divinyl chlorophyll a(1-)"),
    ("CHEBI:73115", "divinyl chlorophyll b"),
    ("CHEBI:73096", "divinyl chlorophyll b(1-)"),
    ("CHEBI:4746", "echinenone"),
    ("CHEBI:16238", "FAD"),
    ("CHEBI:57692", "FAD(3-)"),
    ("CHEBI:60454", "Fe-coproporphyrin III"),
    ("CHEBI:68438", "Fe-coproporphyrin III(4-)"),
    ("CHEBI:60519", "Fe4S2O2 iron-sulfur-oxygen cluster"),
    ("CHEBI:36183", "ferriheme a"),
    ("CHEBI:60532", "ferriheme a(1-)"),
    ("CHEBI:17627", "ferroheme b"),
    ("CHEBI:60344", "ferroheme b(2-)"),
    ("CHEBI:61717", "ferroheme c(2-)"),
    ("CHEBI:60562", "ferroheme c"),
    ("CHEBI:24040", "flavin adenine dinucleotide"),
    ("CHEBI:24041", "flavin mononucleotide"),
    ("CHEBI:17621", "FMN"),
    ("CHEBI:58210", "FMN(3-)"),
    ("CHEBI:16048", "FMNH2"),
    ("CHEBI:57618", "FMNH2(2-)"),
    ("CHEBI:57925", "glutathionate(1-)"),
    ("CHEBI:16856", "glutathione"),
    ("CHEBI:62811", "heme d cis-diol"),
    ("CHEBI:62814", "heme d cis-diol(2-)"),
    ("CHEBI:24480", "heme o"),
    ("CHEBI:16240", "hydrogen peroxide"),
    ("CHEBI:17544", "hydrogencarbonate"),
    ("CHEBI:43474", "hydrogenphosphate"),
    ("CHEBI:17594", "hydroquinone"),
    ("CHEBI:24875", "iron cation"),
    ("CHEBI:60504", "iron-sulfur-iron cofactor"),
    ("CHEBI:30409", "iron-sulfur-molybdenum cofactor"),
    ("CHEBI:60357", "iron-sulfur-vanadium cofactor"),
    ("CHEBI:29033", "iron(2+)"),
    ("CHEBI:29034", "iron(3+)"),
    ("CHEBI:231841", "lanthanum cation"),
    ("CHEBI:49701", "lanthanum(3+)"),
    ("CHEBI:49807", "lead(2+)"),
    ("CHEBI:18420", "magnesium(2+)"),
    ("CHEBI:29035", "manganese(2+)"),
    ("CHEBI:44245", "menaquinone-7"),
    ("CHEBI:28115", "methylcobalamin"),
    ("CHEBI:25372", "molybdopterin cofactor"),
    ("CHEBI:16768", "mycothiol"),
    ("CHEBI:57540", "NAD(1-)"),
    ("CHEBI:15846", "NAD+"),
    ("CHEBI:16908", "NADH"),
    ("CHEBI:57945", "NADH(2-)"),
    ("CHEBI:58349", "NADP(3-)"),
    ("CHEBI:18009", "NADP+"),
    ("CHEBI:16474", "NADPH"),
    ("CHEBI:57783", "NADPH(4-)"),
    ("CHEBI:137373", "Ni(II)-pyridinium-3,5-bisthiocarboxylate mononucleotide(1-)"),
    ("CHEBI:137399", "Ni(II)-pyridinium-3,5-bisthiocarboxylic acid mononucleotide"),
    ("CHEBI:25516", "nickel cation"),
    ("CHEBI:49786", "nickel(2+)"),
    ("CHEBI:17154", "nicotinamide"),
    ("CHEBI:47739", "NiFe4S4 cluster"),
    ("CHEBI:177874", "NiFe4S5 cluster"),
    ("CHEBI:16858", "pantetheine 4'-phosphate"),
    ("CHEBI:47942", "pantetheine 4'-phosphate(2-)"),
    ("CHEBI:25848", "pantothenic acids"),
    ("CHEBI:18067", "phylloquinone"),
    ("CHEBI:26116", "phytochromobilin"),
    ("CHEBI:36079", "polypeptide-derived cofactor"),
    ("CHEBI:26214", "porphyrins"),
    ("CHEBI:29103", "potassium(1+)"),
    ("CHEBI:150862", "premycofactocin"),
    ("CHEBI:87749", "prenyl-FMN"),
    ("CHEBI:87746", "prenyl-FMN(2-)"),
    ("CHEBI:87531", "prenyl-FMNH2"),
    ("CHEBI:17310", "pyridoxal"),
    ("CHEBI:18405", "pyridoxal 5'-phosphate"),
    ("CHEBI:597326", "pyridoxal 5'-phosphate(2-)"),
    ("CHEBI:131529", "pyridoxal hydrochloride"),
    ("CHEBI:16709", "pyridoxine"),
    ("CHEBI:26461", "pyrroloquinoline cofactor"),
    ("CHEBI:18315", "pyrroloquinoline quinone"),
    ("CHEBI:58442", "pyrroloquinoline quinone(3-)"),
    ("CHEBI:15361", "pyruvate"),
    ("CHEBI:32816", "pyruvic acid"),
    ("CHEBI:17015", "riboflavin"),
    ("CHEBI:57986", "riboflavin(1-)"),
    ("CHEBI:59560", "sapropterin"),
    ("CHEBI:28599", "siroheme"),
    ("CHEBI:60052", "siroheme(8-)"),
    ("CHEBI:18023", "sirohydrochlorin"),
    ("CHEBI:58351", "sirohydrochlorin(8-)"),
    ("CHEBI:29101", "sodium(1+)"),
    ("CHEBI:35104", "strontium(2+)"),
    ("CHEBI:16189", "sulfate"),
    ("CHEBI:49883", "tetra-mu3-sulfido-tetrairon"),
    ("CHEBI:71177", "tetrahydromonapterin"),
    ("CHEBI:9532", "thiamine(1+) diphosphate"),
    ("CHEBI:58937", "thiamine(1+) diphosphate(3-)"),
    ("CHEBI:35172", "vanadium cation"),
    ("CHEBI:36970", "vitamin B6 phosphate"),
    ("CHEBI:27547", "zeaxanthin"),
    ("CHEBI:29105", "zinc(2+)"),
    ("CHEBI:17579", "beta-carotene"),
)

_BROAD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("FeS_cluster", ("iron-sulfur", "nife4s", "fe4s", "sulfido-tetrairon", "femo cofactor")),
    ("flavin", ("oxidized flavin",)),
    ("Mo", ("molybdenum",)),
    ("W", ("tungsten",)),
    ("V", ("vanadium",)),
    ("Ba2+", ("barium",)),
    ("Cd2+", ("cadmium",)),
    ("La3+", ("lanthanum",)),
    ("NADP", ("nadp", "nadph")),
    ("NAD", ("nad", "nadh", "nicotinamide adenine dinucleotide")),
    ("FAD", ("fad", "flavin adenine dinucleotide")),
    ("FMN", ("fmn", "riboflavin", "flavin mononucleotide", "prenyl-fmn")),
    ("PLP", ("pyridoxal 5", "pyridoxal phosphate", "vitamin b6 phosphate")),
    ("TPP", ("thiamine",)),
    ("ATP", ("atp", "adenosine 5'-monophosphate", "amp")),
    ("SAM", ("s-adenosyl-l-methionine", "s-adenosyl-l-homocysteine")),
    ("biotin", ("biotin",)),
    ("heme", ("heme", "ferriheme", "ferroheme", "siroheme", "sirohydrochlorin")),
    ("PQQ", ("pyrroloquinoline",)),
    ("quinone", ("quinone", "menaquinone", "phylloquinone", "plastoquinone")),
    ("molybdopterin", ("molybdopterin",)),
    ("THF", ("tetrahydrofolate", "tetrahydrofolic", "methylenetetrahydrofolate")),
    ("lipoate", ("lipoate", "lipoic", "lipoamide")),
    ("cobalamin", ("cobalamin", "cobamide", "cob(i)alamin", "cob(ii)alamin", "corrin", "corrinoid")),
    ("ascorbate", ("ascorbate", "ascorbic")),
    ("pterin", ("pteridine", "sapropterin", "tetrahydromonapterin")),
    ("chlorophyll", ("chlorophyll",)),
    ("carotenoid", ("carotene", "zeaxanthin", "echinenone")),
    ("F430", ("f430",)),
    ("F420", ("f420", "deazaflavin")),
    ("CoM", ("coenzyme m", "mercaptoethanesulfonate")),
    ("CoB", ("coenzyme b", "mercaptoheptanoylthreonine")),
    ("methanofuran", ("methanofuran",)),
    ("methanopterin", ("methanopterin",)),
    ("glutathione", ("glutathione", "glutathionate")),
    ("bacillithiol", ("bacillithiol",)),
    ("mycothiol", ("mycothiol",)),
    ("Mg2+", ("magnesium",)),
    ("Zn2+", ("zinc",)),
    ("Fe2+", ("iron(2+)",)),
    ("Fe3+", ("iron(3+)",)),
    ("Mn2+", ("manganese",)),
    ("Cu2+", ("copper",)),
    ("Ni2+", ("nickel",)),
    ("Ca2+", ("calcium",)),
    ("Co2+", ("cobalt",)),
    ("K+", ("potassium",)),
    ("Na+", ("sodium",)),
    ("Pr3+", ("praseodymium", "pr(3+)", "pr3+")),
    ("Nd3+", ("neodymium", "nd(3+)", "nd3+")),
    ("Eu3+", ("europium", "eu(3+)", "eu3+")),
    ("Sm3+", ("samarium", "sm(3+)", "sm3+")),
    (
        "metal",
        (
            "metal",
            "cerium",
            "lead",
            "strontium",
            "iron cation",
            "divalent metal cation",
            "metal cation",
        ),
    ),
)


def _fold_for_match(text: str) -> str:
    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "μ": "mu",
        "µ": "mu",
        "β": "beta",
        "Β": "beta",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold()


def _strip_charge_suffix(text: str) -> str:
    return re.sub(r"\(\d*[+-]\)$", "", text).strip()


def _exact_label_from_name(name: str) -> str:
    text = _fold_for_match(_strip_charge_suffix(name))
    text = re.sub(r"^\([0-9rs+-]+\)-", "", text)
    text = re.sub(r"[^a-z0-9+]+", "_", text).strip("_")
    return text or "chebi_cofactor"


def _broad_pattern_matches(folded_text: str, pattern: str) -> bool:
    pattern_folded = _fold_for_match(pattern)
    if not pattern_folded:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(pattern_folded)}(?![a-z0-9])", folded_text) is not None


def _broad_label_for_term(chebi_id: str, name: str) -> str:
    folded_name = _fold_for_match(name)
    for label, patterns in _BROAD_RULES:
        if any(_broad_pattern_matches(folded_name, pattern) for pattern in patterns):
            return label
    return _exact_label_from_name(name)


def cofactor_label_for_chebi_term(chebi_id: str, name: str) -> str:
    """Return the controlled cofactor label for a ChEBI cofactor-role filler."""

    return _broad_label_for_term(chebi_id, name)


def _term_aliases(chebi_id: str, name: str) -> list[str]:
    aliases = {chebi_id, name, _strip_charge_suffix(name)}
    folded_name = _fold_for_match(name)
    aliases.add(folded_name)
    aliases.add(_strip_charge_suffix(folded_name))
    aliases.add(re.sub(r"^\([0-9rs+-]+\)-", "", _strip_charge_suffix(folded_name)).strip())
    return sorted(alias for alias in aliases if alias)


def _merge_aliases(target: dict[str, set[str]], label: str, aliases: Iterable[str]) -> None:
    target.setdefault(label, set()).update(str(alias) for alias in aliases if str(alias).strip())


@lru_cache(maxsize=1)
def _default_cofactor_aliases() -> dict[str, list[str]]:
    aliases: dict[str, set[str]] = {}
    for label, values in CURATED_COFACTOR_ALIASES.items():
        _merge_aliases(aliases, label, values)
    for chebi_id, name in CHEBI_COFACTOR_ROLE_TERMS:
        label = _broad_label_for_term(chebi_id, name)
        _merge_aliases(aliases, label, _term_aliases(chebi_id, name))
        if label in {
            "Fe2+",
            "Fe3+",
            "Mg2+",
            "Zn2+",
            "Mn2+",
            "Cu2+",
            "Ni2+",
            "Ca2+",
            "Co2+",
            "K+",
            "Na+",
            "Mo",
            "W",
            "V",
            "Ba2+",
            "Cd2+",
            "La3+",
            "Pr3+",
            "Nd3+",
            "Eu3+",
            "Sm3+",
        }:
            _merge_aliases(aliases, "metal", [chebi_id, name])
    return {label: sorted(values) for label, values in aliases.items()}


# Public compatibility name. This is now ChEBI-expanded, not only curated.
COFACTOR_ALIASES: dict[str, list[str]] = _default_cofactor_aliases()
_SMILES_ALIAS_CACHE: dict[int, tuple[int, dict[str, set[str]], dict[str, set[str]]]] = {}


CORE_COFACTOR_LABELS: frozenset[str] = frozenset(
    {
        "NAD",
        "NADP",
        "FAD",
        "FMN",
        "PLP",
        "TPP",
        "ATP",
        "GTP",
        "SAM",
        "sugar_nucleotide",
        "CoA",
        "biotin",
        "heme",
        "FeS_cluster",
        "THF",
        "lipoate",
        "cobalamin",
        "cobamamide",
        "flavin",
        "quinone",
        "PQQ",
        "molybdopterin",
        "F430",
        "siroheme",
        "F420",
        "CoM",
        "CoB",
        "methanofuran",
        "methanopterin",
        "glutathione",
        "bacillithiol",
        "mycothiol",
        "ascorbate",
        "pterin",
        "chlorophyll",
        "carotenoid",
        NO_COFACTOR_LABEL,
    }
)

METAL_ION_LABELS: frozenset[str] = frozenset(
    {
        "metal",
        "Mg2+",
        "Zn2+",
        "Fe2+",
        "Fe3+",
        "Mn2+",
        "Cu2+",
        "Ni2+",
        "Ca2+",
        "Co2+",
        "K+",
        "Na+",
        "Mo",
        "W",
        "V",
        "Ba2+",
        "Cd2+",
        "La3+",
        "Pr3+",
        "Nd3+",
        "Eu3+",
        "Sm3+",
    }
)

COMMON_AUXILIARY_PARTICIPANTS: frozenset[str] = frozenset(
    {
        "ammonium",
        "chloride",
        "hydrogencarbonate",
        "hydrogen_peroxide",
        "hydrogenphosphate",
        "sulfate",
    }
)


def cofactor_label_tier(label: str) -> str:
    """Classify a cofactor-like label by biological use in enzyme pretraining."""

    text = str(label)
    if text in CORE_COFACTOR_LABELS:
        return "core_cofactor"
    if text in METAL_ION_LABELS:
        return "metal_or_ion"
    return "auxiliary_participant"


def split_cofactor_label_tiers(labels: Iterable[str]) -> dict[str, list[str]]:
    """Split mixed cofactor detections into biologically meaningful tiers."""

    tiers = {
        "core_cofactor_labels": set(),
        "metal_ion_labels": set(),
        "auxiliary_participant_labels": set(),
    }
    for label in labels:
        tier = cofactor_label_tier(str(label))
        if tier == "core_cofactor":
            tiers["core_cofactor_labels"].add(str(label))
        elif tier == "metal_or_ion":
            tiers["metal_ion_labels"].add(str(label))
        else:
            tiers["auxiliary_participant_labels"].add(str(label))
    return {key: sorted(values) for key, values in tiers.items()}


def cofactor_architecture_bins(labels: Iterable[str]) -> list[str]:
    """Map fine cofactor labels to broad motif/domain-oriented bins."""

    bins: set[str] = set()
    seen_label = False
    for label in labels:
        text = str(label)
        if not text:
            continue
        seen_label = True
        if text in METAL_ION_LABELS:
            bins.add("simple_metal_binding")
        bins.update(_COFACTOR_ARCHITECTURE_BIN_MAP.get(text, ()))
    if not bins and not seen_label:
        bins.add(NONE_OR_UNKNOWN_ARCHITECTURE_BIN)
    elif seen_label and not bins:
        bins.add("other_organic_group_transfer")
    return sorted(bins)


def cofactor_chemistry_bins(labels: Iterable[str]) -> list[str]:
    """Map fine cofactor labels to broad chemistry/mechanism bins."""

    bins: set[str] = set()
    seen_label = False
    for label in labels:
        text = str(label)
        if not text:
            continue
        seen_label = True
        if text in METAL_ION_LABELS:
            bins.add("metal_catalysis" if text in _METAL_CATALYTIC_LABELS else "structural_metal")
        bins.update(_COFACTOR_CHEMISTRY_BIN_MAP.get(text, ()))
    if not bins:
        bins.add(UNKNOWN_CHEMISTRY_BIN)
    return sorted(bins)


def cofactor_labels_for_substrate_filter(labels: Iterable[str]) -> set[str]:
    """Labels whose matching molecules should not become substrate/product classes."""

    out = set(CORE_COFACTOR_LABELS) | set(METAL_ION_LABELS) | set(COMMON_AUXILIARY_PARTICIPANTS)
    out.update(str(label) for label in labels if cofactor_label_tier(str(label)) == "core_cofactor")
    return out


def load_cofactor_aliases(path: str | Path | None = None) -> dict[str, list[str]]:
    """Load an optional cofactor dictionary and merge it with built-in aliases.

    Supported inputs:
    - JSON ``{"label": ["alias", ...]}``
    - JSON list of objects with ``label``, ``chebi_id``, ``name``, ``aliases``
    - CSV/TSV with columns ``label`` and any of ``chebi_id``, ``name``,
      ``alias``/``aliases``.
    """

    merged: dict[str, set[str]] = {
        label: set(aliases) for label, aliases in _default_cofactor_aliases().items()
    }
    if path is None:
        return {label: sorted(values) for label, values in merged.items()}

    dictionary_path = Path(path)
    if not dictionary_path.exists():
        raise FileNotFoundError(f"Cofactor dictionary not found: {dictionary_path}")

    def add_record(record: dict[str, Any]) -> None:
        label = str(record.get("label") or record.get("cofactor_label") or "").strip()
        name = str(record.get("name") or record.get("preferred_name") or "").strip()
        chebi_id = str(record.get("chebi_id") or record.get("id") or "").strip()
        if not label:
            label = _broad_label_for_term(chebi_id, name) if name else chebi_id
        elif name:
            inferred_label = _broad_label_for_term(chebi_id, name)
            exact_label = _exact_label_from_name(name)
            if label == exact_label and inferred_label != exact_label:
                label = inferred_label
            elif label == "NAD" and inferred_label != "NAD":
                label = inferred_label
        if not label:
            return
        aliases = []
        if chebi_id:
            aliases.append(chebi_id)
        if name:
            aliases.extend(_term_aliases(chebi_id, name))
        raw_aliases = record.get("aliases") or record.get("alias") or record.get("synonyms") or ""
        if isinstance(raw_aliases, list):
            aliases.extend(str(alias) for alias in raw_aliases)
        else:
            aliases.extend(
                part.strip()
                for part in str(raw_aliases).replace("|", ";").split(";")
                if part.strip()
            )
        raw_smiles = record.get("smiles") or record.get("smiles_string") or ""
        raw_smiles_values = raw_smiles if isinstance(raw_smiles, list) else str(raw_smiles).split("|")
        for smiles in raw_smiles_values:
            if str(smiles).strip():
                aliases.append(f"SMILES_RAW:{_fold_for_match(str(smiles).strip())}")
            canonical = _canonicalize_smiles_alias(str(smiles))
            if canonical:
                aliases.append(f"SMILES:{canonical}")
        _merge_aliases(merged, label, aliases)

    if dictionary_path.suffix.lower() == ".json":
        data = json.loads(dictionary_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for label, aliases in data.items():
                if isinstance(aliases, list):
                    _merge_aliases(merged, str(label), aliases)
        elif isinstance(data, list):
            for record in data:
                if isinstance(record, dict):
                    add_record(record)
        else:
            raise ValueError(f"Unsupported JSON cofactor dictionary format: {dictionary_path}")
    else:
        with dictionary_path.open("r", encoding="utf-8", newline="") as handle:
            if dictionary_path.suffix.lower() == ".tsv":
                dialect = csv.excel_tab
            else:
                sample = handle.read(4096)
                handle.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters="\t,")
                except csv.Error:
                    dialect = csv.excel
            reader = csv.DictReader(handle, dialect=dialect)
            for record in reader:
                add_record(record)

    return {label: sorted(values) for label, values in merged.items()}


def _candidate_smiles_tokens(texts: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for text in texts:
        raw = str(text).strip()
        if not raw:
            continue
        if ">" in raw:
            raw = raw.replace(">>", ".").replace(">", ".")
        if not any(marker in raw for marker in ("=", "#", "(", ")", "[", "]", "@", "\\", "/", ".")):
            continue
        for token in raw.replace(";", ".").replace("|", ".").split("."):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def _mol_from_smiles(smiles: str):
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except Exception:
        return None
    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles)


def _canonicalize_smiles_alias(smiles: str) -> str | None:
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit import Chem
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _dictionary_smiles_labels(
    texts: Iterable[str],
    cofactor_aliases: dict[str, list[str]],
) -> set[str]:
    cache_key = id(cofactor_aliases)
    alias_count = sum(len(aliases) for aliases in cofactor_aliases.values())
    cached = _SMILES_ALIAS_CACHE.get(cache_key)
    if cached is None or cached[0] != alias_count:
        canonical_to_labels: dict[str, set[str]] = {}
        raw_to_labels: dict[str, set[str]] = {}
        for label, aliases in cofactor_aliases.items():
            for alias in aliases:
                if str(alias).startswith("SMILES:"):
                    canonical_to_labels.setdefault(str(alias)[7:], set()).add(label)
                elif str(alias).startswith("SMILES_RAW:"):
                    raw_to_labels.setdefault(str(alias)[11:], set()).add(label)
        _SMILES_ALIAS_CACHE[cache_key] = (alias_count, canonical_to_labels, raw_to_labels)
    else:
        _, canonical_to_labels, raw_to_labels = cached
    if not canonical_to_labels and not raw_to_labels:
        return set()

    labels: set[str] = set()
    for smiles in _candidate_smiles_tokens(texts):
        labels.update(raw_to_labels.get(_fold_for_match(smiles), set()))
        canonical = _canonicalize_smiles_alias(smiles)
        if canonical:
            labels.update(canonical_to_labels.get(canonical, set()))
    return labels


def _has_smarts(mol, smarts: str) -> bool:
    try:
        from rdkit import Chem
    except Exception:
        return False
    query = Chem.MolFromSmarts(smarts)
    return bool(query is not None and mol.HasSubstructMatch(query))


def _atom_count(mol, symbol: str) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == symbol)


def _looks_like_flavin(smiles: str, mol) -> bool:
    compact = smiles.replace("\\", "").replace("/", "")
    if "C1=C(C)C=C2C(=C1)" in compact and "NC(=O)NC" in compact:
        return True
    return _has_smarts(mol, "NC(=O)NC(=O)") and _atom_count(mol, "N") >= 4


def _structural_cofactor_labels(texts: Iterable[str]) -> set[str]:
    labels: set[str] = set()
    for smiles in _candidate_smiles_tokens(texts):
        mol = _mol_from_smiles(smiles)
        if mol is None:
            continue

        local_labels: set[str] = set()
        phosphorus_count = _atom_count(mol, "P")
        sulfur_count = _atom_count(mol, "S")
        has_adenine = _has_smarts(mol, "n1cnc2c(N)ncnc21")
        has_guanine = _has_smarts(mol, "Nc1nc2[nH]cnc2c(=O)[nH]1")
        has_nicotinamide = _has_smarts(mol, "NC(=O)c1ccc[n+]c1") or _has_smarts(
            mol, "NC(=O)C1=CN(C)C=CC1"
        )
        has_coa = _has_smarts(mol, "SCCNC(=O)CCNC(=O)") or _has_smarts(
            mol, "NCCC(=O)NCCS"
        )
        has_sam = _has_smarts(mol, "[S+](C)(CC[C@H]([NH3+])C(=O)[O-])C")
        has_quinone = _has_smarts(mol, "O=C1C=CC(=O)C=C1")

        if has_nicotinamide:
            local_labels.add("NADP" if phosphorus_count >= 3 else "NAD")
        if _looks_like_flavin(smiles, mol):
            local_labels.add("FAD" if has_adenine and phosphorus_count >= 2 else "FMN")
        if has_coa:
            local_labels.add("CoA")
        if has_sam:
            local_labels.add("SAM")
        if has_quinone:
            local_labels.add("quinone")
        if phosphorus_count >= 1 and _has_smarts(mol, "O=CC1=NC=C(COP(=O)(O)O)C(O)=C1"):
            local_labels.add("PLP")

        purine_carrier_labels = {"NAD", "NADP", "FAD", "FMN", "CoA", "SAM"}
        if not (local_labels & purine_carrier_labels):
            if has_adenine and phosphorus_count >= 1:
                local_labels.add("ATP")
            elif has_guanine and phosphorus_count >= 1:
                local_labels.add("GTP")

        atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
        if "Fe" in atom_symbols and sulfur_count >= 2:
            local_labels.add("FeS_cluster")
        for symbol, label in {
            "Mg": "Mg2+",
            "Zn": "Zn2+",
            "Mn": "Mn2+",
            "Cu": "Cu2+",
            "Mo": "Mo",
            "W": "W",
            "V": "V",
            "Ba": "Ba2+",
            "Cd": "Cd2+",
            "La": "La3+",
        }.items():
            if symbol in atom_symbols:
                local_labels.add(label)
                local_labels.add("metal")
        if "Fe" in atom_symbols:
            local_labels.add("metal")

        labels.update(local_labels)
    return labels


def _alias_matches(folded_text: str, alias: str) -> bool:
    alias_folded = _fold_for_match(alias)
    if not alias_folded:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(alias_folded)}(?![a-z0-9])"
    return re.search(pattern, folded_text) is not None


def extract_cofactor_labels(
    texts: list[str] | tuple[str, ...],
    cofactor_aliases: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str], str]:
    """Extract controlled cofactor labels from trusted participant/name strings."""

    labels: set[str] = set()
    used_name_match = False
    joined = " ".join(str(text) for text in texts if text is not None)
    folded = _fold_for_match(joined)
    if not folded.strip():
        return [], [], "unknown"

    alias_map = cofactor_aliases or COFACTOR_ALIASES
    for label, aliases in alias_map.items():
        for alias in aliases:
            if str(alias).startswith(("SMILES:", "SMILES_RAW:")):
                continue
            if _alias_matches(folded, alias):
                labels.add(label)
                used_name_match = True
                break

    structural_labels = _structural_cofactor_labels(texts)
    labels.update(structural_labels)
    dictionary_smiles_labels = _dictionary_smiles_labels(texts, alias_map)
    labels.update(dictionary_smiles_labels)

    flags = []
    if labels and used_name_match:
        flags.append("cofactor_from_name_match")
        if any(str(text).upper().startswith("CHEBI:") for text in texts if text is not None):
            flags.append("cofactor_from_chebi_role_match")
    if structural_labels:
        flags.append("cofactor_from_structure_match")
    if dictionary_smiles_labels:
        flags.append("cofactor_from_chebi_smiles_match")

    if dictionary_smiles_labels:
        status = "chebi_smiles_match"
    elif structural_labels:
        status = "structure_match"
    elif labels and used_name_match:
        status = "weak_name_match"
    else:
        status = "unknown"
    return sorted(labels), flags, status


def extract_cofactor_labels_from_smiles(
    smiles: str,
    cofactor_aliases: dict[str, list[str]] | None = None,
) -> list[str]:
    """Fast cofactor detection for one molecule SMILES.

    This skips free-text alias regex matching. Substrate/product filtering calls
    this function for many individual molecules, where structural signatures and
    exact ChEBI-SMILES dictionary matches are the relevant evidence.
    """

    alias_map = cofactor_aliases or COFACTOR_ALIASES
    labels = set(_structural_cofactor_labels([smiles]))
    labels.update(_dictionary_smiles_labels([smiles], alias_map))
    return sorted(labels)
