"""Explicit configuration wiring for residue-level CIRCE training.

Defaults are listed at their call sites; aliases are resolved centrally. These
builders neither mutate configuration nor import Torch or construct models.
"""

from typing import Any, NamedTuple

from horizyn.config import DotDict


class ReactionChirality(NamedTuple):
    path: str | None
    dim: int
    use: bool
    allow_missing: bool


def _config_first(section, *names: str, default=None):
    for name in names:
        value = section.get(name, None)
        if value is not None:
            return value
    return default


def resolve_reaction_chirality(config: DotDict) -> ReactionChirality:
    """Apply ChIRo/chirality/ChIENN precedence, skipping None but retaining False."""
    reaction_chirality_path = _config_first(
        config.data,
        "reaction_chiro_embeds_path",
        "reaction_chirality_embeds_path",
        "reaction_chienn_embeds_path",
        default=None,
    )
    reaction_chirality_dim = _config_first(
        config.data,
        "reaction_chiro_dim",
        "reaction_chirality_dim",
        "reaction_chienn_dim",
        default=256,
    )
    reaction_use_chirality = _config_first(
        config.model,
        "reaction_use_chiro",
        "reaction_use_chirality",
        "reaction_use_chienn",
        default=_config_first(
            config.data,
            "reaction_use_chiro",
            "reaction_use_chirality",
            "reaction_use_chienn",
            default=True,
        ),
    )
    reaction_allow_missing_chirality = _config_first(
        config.data,
        "reaction_allow_missing_chiro",
        "reaction_allow_missing_chirality",
        "reaction_allow_missing_chienn",
        default=False,
    )
    return ReactionChirality(
        reaction_chirality_path,
        reaction_chirality_dim,
        reaction_use_chirality,
        reaction_allow_missing_chirality,
    )


def _with_defaults(section, *, prefix: str = "", **defaults) -> dict[str, Any]:
    """Forward listed fields, optionally prefixing output keys; preserve explicit None."""
    return {prefix + key: section.get(key, default) for key, default in defaults.items()}


def reaction_data_module_kwargs(
    config: DotDict,
    *,
    replay_config: dict[str, Any] | None = None,
    replay_fraction: float = 0.0,
    replay_seed: int | None = None,
) -> dict[str, Any]:
    """Build data options; only externally loaded replay state is passed separately."""
    chirality = resolve_reaction_chirality(config)
    validation_enabled = bool(config.training.get("validation_enabled", True))
    if replay_seed is None:
        replay_seed = int(config.get("seed", 42))
    return dict(
        indexed_pairs_dir=config.data.get("indexed_pairs_dir"),
        train_pairs_path=config.data.train_pairs_path,
        test_pairs_path=config.data.get(
            "validation_pairs_path",
            config.data.get("test_pairs_path", config.data.train_pairs_path),
        ),
        train_reactions_path=config.data.train_reactions_path,
        test_reactions_path=config.data.get(
            "validation_reactions_path",
            config.data.get("test_reactions_path", config.data.train_reactions_path),
        ),
        protein_residue_embeds_path=config.data.protein_residue_embeds_path,
        **_with_defaults(
            config.data,
            protein_score_residue_embeds_path=None,
            cached_enzyme_base_embeds_path=None,
            cached_train_reaction_base_embeds_path=None,
            cached_validation_reaction_base_embeds_path=None,
            protein_functional_tokens_path=None,
        ),
        train_batch_size=config.data.train_batch_size,
        **_with_defaults(
            config.data,
            retrieval_batch_size=1,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            worker_num_threads=None,
            rdkit_fp_dim=1024,
            drfp_dim=1024,
            reaction_representation="fingerprint",
            reaction_embeds_path=None,
            reaction_t5v2_embeds_path=None,
            reaction_model_embeds_path=None,
            reaction_unimol2_embeds_path=None,
            reaction_chiro_embeds_path=None,
            reaction_chirality_embeds_path=None,
        ),
        reaction_chienn_embeds_path=chirality.path,
        **_with_defaults(
            config.data,
            train_reaction_embeds_path=None,
            validation_reaction_embeds_path=None,
            train_reaction_t5v2_embeds_path=None,
            validation_reaction_t5v2_embeds_path=None,
            train_reaction_model_embeds_path=None,
            validation_reaction_model_embeds_path=None,
            train_reaction_unimol2_embeds_path=None,
            validation_reaction_unimol2_embeds_path=None,
            train_reaction_chiro_embeds_path=None,
            validation_reaction_chiro_embeds_path=None,
            train_reaction_chirality_embeds_path=None,
            validation_reaction_chirality_embeds_path=None,
            train_reaction_chienn_embeds_path=None,
            validation_reaction_chienn_embeds_path=None,
            reaction_chemistry_vectors_path=None,
            train_reaction_chemistry_vectors_path=None,
            validation_reaction_chemistry_vectors_path=None,
            reaction_directional_vectors_path=None,
            train_reaction_directional_vectors_path=None,
            validation_reaction_directional_vectors_path=None,
            reaction_model_dim=None,
            reaction_unimol_dim=768,
            reaction_chiro_dim=None,
            reaction_chirality_dim=None,
        ),
        reaction_chienn_dim=chirality.dim,
        **_with_defaults(
            config.data,
            reaction_chemistry_dim=None,
            reaction_directional_dim=None,
            reaction_use_model=True,
            reaction_use_chiro=None,
            reaction_use_chirality=None,
        ),
        reaction_use_chienn=chirality.use,
        **_with_defaults(
            config.data,
            reaction_use_chemistry=False,
            reaction_use_directional=False,
            reaction_load_directional=False,
            reaction_allow_missing_unimol2=False,
            reaction_allow_missing_chiro=None,
            reaction_allow_missing_chirality=None,
        ),
        reaction_allow_missing_chienn=config.data.get(
            "reaction_allow_missing_chienn",
            chirality.allow_missing,
        ),
        **_with_defaults(
            config.data,
            reaction_allow_missing_chemistry=True,
            reaction_allow_missing_directional=True,
            reaction_embedding_in_memory=True,
            residue_dim=1024,
            score_residue_dim=None,
            max_protein_tokens=1024,
            protein_truncation="ends_center",
            standardize_reactions=True,
            standardize_hypervalent=True,
            standardize_remove_hs=True,
            standardize_kekulize=False,
            standardize_uncharge=True,
            standardize_metals=True,
            normalize_molecule_sets_as_self_reactions=False,
            enzyme_ec_labels_path=None,
            protein_capability_vectors_path=None,
            protein_capability_metadata_path=None,
            capability_vector_in_memory=True,
        ),
        capability_missing_policy=config.data.get(
            "capability_missing_policy",
            config.model.get("capability_vector", {}).get("missing_policy", "zero_with_mask"),
        ),
        protein_factorized_capability_vectors_path=config.data.get(
            "protein_factorized_capability_vectors_path",
            None,
        ),
        factorized_capability_missing_policy=config.data.get(
            "factorized_capability_missing_policy",
            config.model.get("factorized_capability_vector", {}).get(
                "missing_policy", "zero_with_mask"
            ),
        ),
        **_with_defaults(
            config.data, protein_text_vectors_path=None, protein_text_metadata_path=None
        ),
        text_vector_missing_policy=config.data.get(
            "text_vector_missing_policy",
            config.model.get("text_vector", {}).get("missing_policy", "zero_with_mask"),
        ),
        **_with_defaults(
            config.data, protein_biofp_targets_path=None, protein_biofp_vocab_path=None
        ),
        biofp_missing_policy=config.data.get(
            "biofp_missing_policy",
            config.model.get("biofp", {}).get("missing_policy", "zero_with_mask"),
        ),
        validation_retrieval_metrics=config.training.get("validation_retrieval_metrics", False),
        validation_retrieval_candidate_set=config.training.get(
            "validation_retrieval_candidate_set",
            config.data.get("validation_retrieval_candidate_set", "validation"),
        ),
        validation_retrieval_candidate_ids_path=config.training.get(
            "validation_retrieval_candidate_ids_path",
            config.data.get("validation_retrieval_candidate_ids_path"),
        ),
        validation_retrieval_batch_size=config.training.get(
            "validation_retrieval_batch_size",
            config.data.get(
                "validation_retrieval_batch_size", config.data.get("retrieval_batch_size", 1)
            ),
        ),
        validation_retrieval_query_ids_path=config.data.get("validation_retrieval_query_ids_path"),
        validation_retrieval_directions=config.training.get(
            "validation_retrieval_directions",
            config.data.get("validation_retrieval_directions"),
        ),
        **_with_defaults(
            config.data,
            hard_negative_pools_path=None,
            hard_negative_direction="reaction_to_enzyme",
            hard_negative_anchor_queries_per_batch=48,
            hard_negative_positives_per_query=2,
            hard_negative_negatives_per_query=8,
        ),
        hard_negative_seed=config.data.get("hard_negative_seed", config.get("seed", 42)),
        typed_negative_pools_path=config.data.get("typed_negative_pools_path"),
        typed_negative_positive_fraction=config.data.get(
            "typed_negative_positive_fraction",
            0.85 if config.data.get("indexed_pairs_dir") else 0.5,
        ),
        typed_negative_biological_fraction=config.data.get(
            "typed_negative_biological_fraction",
            0.5,
        ),
        typed_negative_seed=config.data.get("typed_negative_seed", config.get("seed", 42)),
        reaction_balanced_sampling=config.data.get("train_sampler", {}).get("name", "shuffle")
        == "reaction_degree_balanced",
        balanced_anchor_sampling=config.data.get("train_sampler", {}).get("name") == "balanced_anchors",
        balanced_anchor_positives=config.data.get("train_sampler", {}).get("positives_per_anchor", 4),
        balanced_anchor_seed=config.data.get("train_sampler", {}).get("seed", config.get("seed", 42)),
        reaction_balanced_degree_exponent=config.data.get("train_sampler", {}).get(
            "degree_exponent",
            0.5,
        ),
        reaction_balanced_seed=config.data.get("train_sampler", {}).get(
            "seed",
            config.get("seed", 42),
        ),
        enzyme_grouped_sampling=config.data.get("train_sampler", {}).get("name", "shuffle")
        == "enzyme_grouped",
        **_with_defaults(
            config.data.get("train_sampler", {}),
            prefix="enzyme_grouped_",
            anchors_per_batch=64,
            positives_per_anchor=4,
        ),
        enzyme_grouped_seed=config.data.get("train_sampler", {}).get(
            "seed",
            config.get("seed", 42),
        ),
        hypergraph_sampling=config.data.get("train_sampler", {}).get("name", "shuffle")
        == "hypergraph_grouped",
        **_with_defaults(
            config.data.get("train_sampler", {}),
            prefix="hypergraph_",
            anchors_per_batch=16,
            positives_per_anchor=4,
            reaction_anchor_fraction=0.5,
        ),
        hypergraph_seed=config.data.get("train_sampler", {}).get("seed", config.get("seed", 42)),
        replay_config=replay_config,
        replay_fraction=replay_fraction,
        replay_seed=replay_seed,
        reaction_direction_mode=config.data.get("reaction_direction_mode", "bidirectional"),
        validation_enabled=validation_enabled,
    )


def _experimental_head_kwargs(config: DotDict) -> dict[str, Any]:
    """Legacy adapters, prototypes and biological residual heads (all still supported).

    Include inactive defaults too: constructor and checkpoint schemas depend on them.
    """
    enzyme_block_config = config.model.get("enzyme_block_fusion", {})
    enzyme_prototype_config = config.model.get("enzyme_prototypes", {})
    biological_residual_config = config.model.get("biological_residual", {})
    e2r_adapter_config = config.model.get("e2r_adapter", {})
    r2e_adapter_config = config.model.get("r2e_adapter", {})
    return dict(
        **_with_defaults(
            e2r_adapter_config,
            prefix="e2r_adapter_",
            enabled=False,
            hidden_dim=512,
            dropout=0.1,
            gate_init=0.1,
            use_factorized_inputs=False,
            use_directional_inputs=False,
        ),
        e2r_adapter_directional_dim=e2r_adapter_config.get(
            "directional_dim",
            config.data.get("reaction_directional_dim"),
        ),
        e2r_adapter_directional_hidden_dim=e2r_adapter_config.get("directional_hidden_dim", 128),
        e2r_identity_weight=config.training.loss.get("e2r_identity_weight", 0.0),
        **_with_defaults(
            r2e_adapter_config,
            prefix="r2e_adapter_",
            enabled=False,
            hidden_dim=512,
            dropout=0.1,
            gate_init=0.1,
            use_factorized_inputs=False,
        ),
        r2e_adapter_block_dims=r2e_adapter_config.get(
            "block_dims",
            enzyme_block_config.get("dims"),
        ),
        r2e_adapter_block_weights=r2e_adapter_config.get(
            "block_weights",
            enzyme_block_config.get("weights"),
        ),
        r2e_identity_weight=config.training.loss.get("r2e_identity_weight", 0.0),
        **_with_defaults(
            enzyme_prototype_config,
            prefix="enzyme_prototype_",
            count=1,
            bottleneck_dim=64,
            residual_gate_init=0.05,
            temperature=0.1,
            dropout=0.0,
            routing_entropy_weight=0.0,
            load_balance_weight=0.0,
        ),
        **_with_defaults(
            biological_residual_config,
            prefix="biological_residual_",
            enabled=False,
            token_dim=128,
            heads=4,
            layers=1,
            dropout=0.1,
            max_molecules=32,
            sleec_bias=0.25,
            sleec_pool_scale=2.0,
            lse_temperature=0.1,
            target_chunk_size=128,
            fusion_alpha=0.0,
            max_fusion_alpha=0.2,
        ),
        biological_residual_base_loss_weight=config.training.loss.get(
            "biological_base_weight",
            0.0,
        ),
        biological_residual_fused_loss_weight=config.training.loss.get(
            "biological_fused_weight",
            0.0,
        ),
        biological_residual_local_loss_weight=config.training.loss.get(
            "biological_local_weight",
            1.0,
        ),
        biological_residual_guard_weight=config.training.loss.get("biological_guard_weight", 0.1),
        biological_residual_penalty_weight=config.training.loss.get(
            "biological_penalty_weight",
            0.01,
        ),
    )


def protein_pooling_model_kwargs(config: DotDict) -> dict[str, Any]:
    """Build all model, optimizer and loss options from a single configuration."""
    chirality = resolve_reaction_chirality(config)
    pooling_config = config.model.get("protein_attention_pooling", {})
    sleec_config = config.model.get("sleec_pooling", {})
    enzyme_fusion_config = config.model.get("enzyme_fusion", {})
    capability_config = config.model.get("capability_vector", {})
    factorized_capability_config = config.model.get("factorized_capability_vector", {})
    biofp_config = config.model.get("biofp", {})
    text_config = config.model.get("text_vector", {})
    hyperbolic_config = config.model.get("hyperbolic_encoder", {})
    enzyme_block_config = config.model.get("enzyme_block_fusion", {})
    reaction_attention_regularization_config = config.training.get(
        "reaction_attention_regularization",
        {},
    )
    reaction_chemistry_consistency_config = config.training.get(
        "reaction_chemistry_consistency",
        {},
    )
    reaction_residual_identity_config = config.training.get(
        "reaction_residual_identity",
        {},
    )
    reaction_hyperbolic_config = config.model.get("reaction_hyperbolic_encoder", {})
    reaction_multimodal_attention_config = config.model.get(
        "reaction_multimodal_attention",
        {},
    )
    symmetric_reaction_block_config = reaction_multimodal_attention_config.get(
        "symmetric_output_blocks",
        {},
    )
    symmetric_reaction_blocks_enabled = symmetric_reaction_block_config.get(
        "enabled",
        False,
    )
    reaction_fingerprint_attention_config = config.model.get(
        "reaction_fingerprint_attention",
        {},
    )
    query_encoder_checkpoint_path = config.model.get("query_encoder_checkpoint_path", None)
    return dict(
        query_encoder_dims=config.model.query_encoder_dims,
        target_encoder_dims=config.model.target_encoder_dims,
        embedding_dim=config.model.embedding_dim,
        residue_dim=config.data.get("residue_dim", 1024),
        pooling=config.model.get("pooling", "mean"),
        attention_bias=pooling_config.get("attention_bias", True),
        **_with_defaults(
            sleec_config,
            prefix="sleec_",
            mode="topk",
            topk_fraction=0.2,
            threshold=0.5,
            scorer_hidden_dim=256,
        ),
        sleec_score_hidden_dim=sleec_config.get(
            "score_hidden_dim",
            config.data.get("score_residue_dim"),
        ),
        **_with_defaults(sleec_config, prefix="sleec_", checkpoint_path=None, freeze_scorer=False),
        **_with_defaults(
            sleec_config, prefix="sleec_guided_", initial_bias_scale=1.0, train_bias_scale=True
        ),
        **_with_defaults(
            hyperbolic_config,
            prefix="hyperbolic_",
            checkpoint_path=None,
            freeze_projector=False,
            use_tangent=True,
            hyp_dim=None,
            load_attention_pooler=False,
            freeze_attention_pooler=False,
        ),
        enzyme_input_mode=config.model.get("enzyme_input_mode", "standard"),
        enzyme_multiview=config.model.get("enzyme_multiview", None),
        enzyme_attention_regularization=config.training.get("enzyme_attention_regularization", None),
        **_with_defaults(
            enzyme_fusion_config, prefix="enzyme_fusion_", hidden_dim=None, dropout=0.0
        ),
        **_with_defaults(
            enzyme_block_config,
            prefix="enzyme_block_",
            dims=None,
            weights=None,
            dropout=0.0,
            learned_weights=False,
        ),
        capability_vector_dim=capability_config.get(
            "dim",
            config.model.get("capability_vector_dim", 256),
        ),
        capability_freeze=capability_config.get(
            "freeze",
            config.model.get("capability_freeze", True),
        ),
        capability_adapter=capability_config.get(
            "adapter",
            config.model.get("capability_adapter", False),
        ),
        capability_dropout=capability_config.get(
            "dropout",
            config.model.get("capability_dropout", 0.1),
        ),
        **_with_defaults(
            factorized_capability_config,
            prefix="factorized_capability_",
            dims=None,
            freeze=True,
            use_masks=True,
        ),
        **_with_defaults(
            biofp_config,
            prefix="biofp_",
            center_dim=0,
            cofactor_dim=0,
            transition_dim=0,
            family_dims=None,
            positive_labels=None,
            seq_dim=384,
            dim=128,
            hidden_dim=512,
            seq_weight=0.75,
            dropout=0.1,
        ),
        **_with_defaults(
            config.training.loss,
            biofp_aux_weight=0.0,
            biofp_aux_mode="bce",
            biofp_aux_warmup_epochs=0,
            biofp_normalize_active_families=False,
            biofp_center_weight=0.45,
            biofp_transition_weight=0.35,
            biofp_cofactor_weight=0.2,
            biofp_family_weights=None,
            biofp_confidence_cap=8.0,
            cross_tower_alignment_weight=0.0,
            cross_tower_alignment_family_weights=None,
        ),
        reaction_attention_entropy_weight=reaction_attention_regularization_config.get(
            "weight",
            0.0,
        ),
        reaction_attention_min_normalized_entropy=reaction_attention_regularization_config.get(
            "min_normalized_entropy",
            0.75,
        ),
        reaction_chemistry_consistency_weight=reaction_chemistry_consistency_config.get(
            "weight",
            0.0,
        ),
        reaction_geometry_weight=config.training.get("reaction_geometry", {}).get("weight", 0.0),
        protein_geometry_weight=config.training.get("protein_geometry", {}).get("weight", 0.0),
        biological_geometry_path=config.training.get('biological_geometry', {}).get('path'),
        biological_geometry_weight=config.training.get('biological_geometry', {}).get('weight', 0.0),
        biological_geometry_mode=config.training.get('biological_geometry', {}).get('mode', 'attraction'),
        biological_geometry_margin=config.training.get('biological_geometry', {}).get('margin', 0.1),
        reaction_residual_identity_weight=reaction_residual_identity_config.get("weight", 0.0),
        enzyme_block_weight_kl_weight=enzyme_block_config.get("weight_kl_weight", 0.0),
        text_vector_dim=text_config.get("dim", config.model.get("text_vector_dim", 768)),
        text_fusion_dim=text_config.get("fusion_dim", config.model.get("text_fusion_dim", 512)),
        text_num_heads=text_config.get("num_heads", config.model.get("text_num_heads", 8)),
        text_dropout=text_config.get("dropout", config.model.get("text_dropout", 0.1)),
        text_freeze=text_config.get("freeze", config.model.get("text_freeze", True)),
        text_adapter=text_config.get("adapter", config.model.get("text_adapter", False)),
        **_with_defaults(
            reaction_hyperbolic_config,
            prefix="reaction_hyperbolic_",
            checkpoint_path=None,
            hyp_dim=None,
            freeze_encoder=True,
            freeze_projector=True,
            use_tangent=True,
        ),
        reaction_fingerprint_attention_enabled=reaction_fingerprint_attention_config.get(
            "enabled",
            False,
        ),
        reaction_fingerprint_attention_input_dim=config.data.get("rdkit_fp_dim", 1024)
        + config.data.get("drfp_dim", 1024),
        reaction_fingerprint_attention_rdkit_dim=config.data.get("rdkit_fp_dim", 1024),
        reaction_fingerprint_attention_drfp_dim=config.data.get("drfp_dim", 1024),
        **_with_defaults(
            reaction_fingerprint_attention_config,
            prefix="reaction_fingerprint_attention_",
            token_dim=512,
            hidden_dim=512,
            dropout=0.0,
        ),
        reaction_fingerprint_attention_bias=reaction_fingerprint_attention_config.get(
            "attention_bias",
            True,
        ),
        **_experimental_head_kwargs(config),
        query_normalise_output=config.model.get(
            "query_normalise_output",
            config.model.get("normalise_output", True),
        ),
        target_normalise_output=config.model.get(
            "target_normalise_output",
            config.model.get("normalise_output", True),
        ),
        enforce_normalisation=config.model.get(
            "enforce_normalisation",
            config.model.get("normalise_output", True),
        ),
        embedding_similarity=config.training.get(
            "embedding_similarity",
            config.training.loss.get("embedding_similarity", "cosine"),
        ),
        validation_similarity=config.training.get(
            "validation_similarity",
            config.training.get(
                "embedding_similarity", config.training.loss.get("embedding_similarity", "cosine")
            ),
        ),
        query_encoder_type=config.model.get("query_encoder_type", "mlp"),
        **_with_defaults(config.data, reaction_model_dim=1024, reaction_unimol_dim=768),
        reaction_chienn_dim=chirality.dim,
        **_with_defaults(
            config.data,
            reaction_chemistry_dim=None,
            reaction_directional_dim=None,
            reaction_use_model=True,
        ),
        reaction_use_chienn=chirality.use,
        **_with_defaults(config.data, reaction_use_chemistry=False, reaction_use_directional=False),
        **_with_defaults(
            config.model, reaction_chirality_name="chiro", reaction_pooling="attention"
        ),
        **_with_defaults(
            config.model.get("reaction_attention_pooling", {}),
            prefix="reaction_",
            attention_bias=True,
            separate_side_poolers=True,
        ),
        **_with_defaults(
            reaction_multimodal_attention_config,
            prefix="reaction_modality_attention_",
            hidden_dim=None,
            dropout=0.0,
        ),
        **_with_defaults(
            reaction_multimodal_attention_config,
            prefix="reaction_",
            modality_dropout=0.0,
            chemistry_dropout=0.0,
        ),
        reaction_modality_token_layer_norm=reaction_multimodal_attention_config.get(
            "token_layer_norm",
            reaction_multimodal_attention_config.get("token_normalization", False),
        ),
        **_with_defaults(
            reaction_multimodal_attention_config,
            prefix="reaction_",
            modality_l2_normalize=False,
            modality_encoder_num_layers=None,
            modality_encoder_widths=None,
        ),
        reaction_modality_encoder_use_layer_norm=reaction_multimodal_attention_config.get(
            "modality_encoder_use_layer_norm",
            reaction_multimodal_attention_config.get("modality_encoder_layer_norm", False),
        ),
        **_with_defaults(
            reaction_multimodal_attention_config,
            prefix="reaction_",
            modality_encoder_dropout=0.0,
            modality_encoder_normalise_output=False,
            side_composition="directional_delta",
        ),
        reaction_modality_fusion=reaction_multimodal_attention_config.get("fusion", "attention"),
        **_with_defaults(
            reaction_multimodal_attention_config,
            prefix="reaction_",
            factorized_dims=None,
            factorized_weights=None,
            attention_prior_weights=None,
            attention_adaptation_strength=0.4,
            output_projection="mlp",
        ),
        reaction_output_block_dims=(
            enzyme_block_config.get("dims") if symmetric_reaction_blocks_enabled else None
        ),
        reaction_output_block_weights=(
            enzyme_block_config.get("weights") if symmetric_reaction_blocks_enabled else None
        ),
        reaction_output_block_dropout=symmetric_reaction_block_config.get("dropout", 0.0),
        **_with_defaults(
            reaction_multimodal_attention_config,
            prefix="reaction_",
            residual_gate_init=0.1,
            directional_gate_init=0.1,
        ),
        query_encoder_checkpoint_path=query_encoder_checkpoint_path,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        beta=config.training.loss.beta,
        learn_beta=config.training.loss.get("learn_beta", False),
        beta_min=config.training.loss.get("beta_min", -float("inf")),
        beta_max=config.training.loss.get("beta_max", float("inf")),
        loss_name=config.training.loss.get("name", "FullBatchMLNCELoss"),
        **_with_defaults(
            config.training.loss,
            sampled_require_both_directions=False,
            sampled_share_indexed_negatives=False,
        ),
        validation_retrieval_candidate_chunk_size=config.training.get(
            "validation_retrieval_candidate_chunk_size",
            0,
        ),
        **_with_defaults(
            config.training.loss,
            positive_pair_source="observed_pairs",
            degree_alpha=0.5,
            cardinality_weight=0.3,
            cardinality_warmup_epochs=5,
            separate_direction_temperatures=False,
        ),
        beta_r2e=config.training.loss.get("beta_r2e", config.training.loss.beta),
        beta_e2r=config.training.loss.get("beta_e2r", config.training.loss.beta),
        **_with_defaults(
            config.training.loss,
            temperature_regularization_weight=0.0,
            unknown_negative_weight=1.0,
            soft_rank_weight=0.0,
            soft_rank_tau=0.1,
            soft_rank_top_k=128,
            sigmoid_bias_init=0.0,
            sigmoid_learn_bias=True,
            sigmoid_negative_weight=1.0,
            lambda_r=0.05,
            lambda_e=0.05,
            lambda_g=0.01,
            tau_r=0.1,
            tau_e=0.1,
            tau_t=0.1,
            delta_r=0.5,
            delta_e=0.5,
            symmetric_gw=True,
            direction_balance_weight=0.0,
            lambda_rr=0.0,
            lambda_ee=0.0,
        ),
        lambda_gw=config.training.loss.get("lambda_gw", config.training.loss.get("lambda_g", 0.0)),
        lambda_direction_gap=config.training.loss.get(
            "lambda_direction_gap",
            config.training.loss.get("direction_balance_weight", 0.0),
        ),
        **_with_defaults(
            config.training.loss,
            lambda_r2e=0.5,
            lambda_e2r=0.5,
            lambda_r2e_hard_neg=0.0,
            r2e_hard_neg_top_k=0,
            r2e_hard_neg_margin=0.0,
            lambda_e2r_hard_neg=0.0,
            e2r_hard_neg_top_k=0,
            e2r_hard_neg_margin=0.0,
        ),
        tau_rr=config.training.loss.get("tau_rr", config.training.loss.get("tau_r", 0.1)),
        tau_ee=config.training.loss.get("tau_ee", config.training.loss.get("tau_e", 0.1)),
        tau_gw=config.training.loss.get("tau_gw", config.training.loss.get("tau_t", 0.1)),
        **_with_defaults(
            config.training.loss,
            ec_positive_policy="hierarchical_weighted",
            ec_min_shared_depth=2,
            gw_max_anchors=512,
            apply_structure_terms_on_val=False,
        ),
        **_with_defaults(
            config.training,
            lambda_residue=0.0,
            fused_adamw=False,
            contrastive_fp32=False,
            fp32_sensitive_modules=False,
            log_attention_stats=True,
            attention_logging_interval=100,
            validation_retrieval_metrics=False,
        ),
        validation_retrieval_directions=config.training.get(
            "validation_retrieval_directions",
            config.data.get("validation_retrieval_directions"),
        ),
        retrieval_metric_top_k=config.training.get("metrics", {}).get("top_k", [1, 10, 100, 1000]),
        training_stage=config.training.get("training_stage", "joint"),
        **_with_defaults(
            config.training.loss,
            detach_reaction_embeddings=False,
            anchor_weight=0.0,
            capability_consistency_weight=0.0,
        ),
    )
