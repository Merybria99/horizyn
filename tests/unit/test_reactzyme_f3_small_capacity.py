import copy
from pathlib import Path

import yaml

from horizyn.benchmarks.reactzyme_f3_small_capacity import (
    DEFAULT_SOURCE_CONFIG,
    EXPECTED_REACTION_ENCODER_PARAMETERS,
    MODALITY_ENCODER_WIDTHS,
    QUERY_ENCODER_DIMS,
    configure_small_f3,
    generate,
)
from horizyn.config import DotDict, validate_config
from horizyn.model import MultimodalReactionAttentionEncoder


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_small_f3_changes_only_reaction_hidden_capacity(tmp_path):
    base = _load(DEFAULT_SOURCE_CONFIG)
    config = configure_small_f3(base, run_root=tmp_path)

    expected_model = copy.deepcopy(base["model"])
    expected_model["query_encoder_dims"] = QUERY_ENCODER_DIMS
    expected_model["reaction_multimodal_attention"][
        "modality_encoder_widths"
    ] = MODALITY_ENCODER_WIDTHS

    assert config["model"] == expected_model
    assert config["data"] == base["data"]
    assert config["training"] == base["training"]
    assert config["model"]["embedding_dim"] == 512
    validate_config(DotDict(config))


def test_small_f3_reaction_encoder_parameter_count():
    encoder = MultimodalReactionAttentionEncoder(
        input_dim=512,
        output_dim=512,
        num_layers=1,
        widths=[1024],
        reaction_model_dim=768,
        unimol_dim=768,
        chienn_dim=256,
        reaction_chemistry_dim=617,
        use_reaction_model=True,
        use_chienn=True,
        use_reaction_chemistry=True,
        chirality_modality_name="chiro",
        reaction_pooling="attention",
        attention_bias=True,
        separate_side_poolers=True,
        modality_attention_hidden_dim=512,
        modality_token_layer_norm=True,
        modality_encoder_widths=MODALITY_ENCODER_WIDTHS,
        side_composition="molecule_set",
        modality_fusion="attention",
        output_projection="mlp",
    )

    assert sum(parameter.numel() for parameter in encoder.parameters()) == (
        EXPECTED_REACTION_ENCODER_PARAMETERS
    )


def test_generate_writes_reproducible_manifest_and_config(tmp_path):
    manifest = generate(run_root=tmp_path)

    config_path = Path(manifest["config"])
    assert config_path.is_file()
    assert manifest["embedding_dim"] == 512
    assert manifest["expected_reaction_encoder_parameters"] == 6_937_093
