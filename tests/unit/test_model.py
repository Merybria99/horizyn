"""Tests for model architecture components."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from horizyn.model import (
    MLP,
    BaseModel,
    DualContrastiveModel,
    NormalizeLayer,
    ProteinPooledDualModel,
    TigerTextGatedFusion,
)


class TestNormalizeLayer:
    """Tests for NormalizeLayer."""

    def test_normalization_l2_norm(self, device):
        """Test that L2 normalization produces unit vectors."""
        layer = NormalizeLayer(p=2, dim=-1).to(device)
        x = torch.randn(10, 512, device=device)
        output = layer(x)

        # Check that output has unit L2 norm
        norms = torch.norm(output, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_normalization_shape_preserved(self, device):
        """Test that normalization preserves tensor shape."""
        layer = NormalizeLayer().to(device)
        x = torch.randn(8, 256, device=device)
        output = layer(x)
        assert output.shape == x.shape

    def test_normalization_different_dims(self, device):
        """Test normalization along different dimensions."""
        x = torch.randn(4, 8, 16, device=device)

        # Normalize along last dim
        layer_last = NormalizeLayer(dim=-1).to(device)
        output_last = layer_last(x)
        norms_last = torch.norm(output_last, p=2, dim=-1)
        assert torch.allclose(norms_last, torch.ones_like(norms_last), atol=1e-6)

        # Normalize along first dim
        layer_first = NormalizeLayer(dim=0).to(device)
        output_first = layer_first(x)
        norms_first = torch.norm(output_first, p=2, dim=0)
        assert torch.allclose(norms_first, torch.ones_like(norms_first), atol=1e-6)

    def test_extra_repr(self):
        """Test string representation of layer."""
        layer = NormalizeLayer(p=2, dim=-1, eps=1e-12)
        repr_str = layer.extra_repr()
        assert "p=2" in repr_str
        assert "dim=-1" in repr_str
        assert "eps=" in repr_str

    def test_zero_vector_stability(self, device):
        """Test that zero vectors don't cause numerical issues."""
        layer = NormalizeLayer(eps=1e-12).to(device)
        x = torch.zeros(5, 100, device=device)
        # Should not raise an error due to eps
        output = layer(x)
        assert output.shape == x.shape


class TestBaseModel:
    """Tests for BaseModel."""

    def test_initialization(self):
        """Test that BaseModel initializes correctly."""
        model = BaseModel()
        assert isinstance(model.pre_nn_layers, nn.ModuleList)
        assert isinstance(model.main_nn, nn.ModuleList)
        assert isinstance(model.post_nn_layers, nn.ModuleList)
        assert isinstance(model.output_heads, nn.ModuleDict)
        assert len(model.pre_nn_layers) == 0
        assert len(model.main_nn) == 0
        assert len(model.post_nn_layers) == 0

    def test_extra_args_raises_error(self):
        """Test that extra arguments raise ValueError."""
        with pytest.raises(ValueError, match="Extra unused arguments"):
            BaseModel(extra_arg="value")

        with pytest.raises(ValueError, match="Extra unused arguments"):
            BaseModel("positional_arg")

    def test_model_body_property(self):
        """Test model_body property returns all layers."""
        model = BaseModel()
        model.pre_nn_layers.append(nn.Linear(10, 20))
        model.main_nn.append(nn.ReLU())
        model.post_nn_layers.append(nn.Linear(20, 10))

        body = model.model_body
        assert len(body) == 3

    def test_num_parameters(self):
        """Test parameter counting."""
        model = BaseModel()
        # Empty model has no parameters
        assert model.num_parameters == 0

        # Add a linear layer
        model.main_nn.append(nn.Linear(10, 5))
        # Parameters: 10*5 weights + 5 biases = 55
        assert model.num_parameters == 55

    def test_forward_pass(self, device):
        """Test forward pass through model body."""
        model = BaseModel()
        model.main_nn.append(nn.Linear(10, 20))
        model.main_nn.append(nn.ReLU())
        model.main_nn.append(nn.Linear(20, 5))
        model = model.to(device)  # Move to device after adding layers

        x = torch.randn(4, 10, device=device)
        output = model(x)
        assert output.shape == (4, 5)

    def test_output_heads(self, device):
        """Test that output heads produce dict output."""
        model = BaseModel()
        model.main_nn.append(nn.Linear(10, 20))
        model.output_heads["head1"] = nn.Linear(20, 5)
        model.output_heads["head2"] = nn.Linear(20, 3)
        model = model.to(device)  # Move to device after adding layers

        x = torch.randn(4, 10, device=device)
        output = model(x)

        assert isinstance(output, dict)
        assert "head1" in output
        assert "head2" in output
        assert output["head1"].shape == (4, 5)
        assert output["head2"].shape == (4, 3)


class TestMLP:
    """Tests for MLP model."""

    def test_simple_mlp(self, device):
        """Test basic MLP creation and forward pass."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=1, widths=20).to(device)
        x = torch.randn(4, 10, device=device)
        output = mlp(x)
        assert output.shape == (4, 5)

    def test_two_layer_mlp(self, device):
        """Test MLP with 2 hidden layers."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=2, widths=[20, 15]).to(device)
        x = torch.randn(4, 10, device=device)
        output = mlp(x)
        assert output.shape == (4, 5)

    def test_mlp_with_normalization(self, device):
        """Test MLP with output normalization."""
        mlp = MLP(input_dim=10, output_dim=512, num_layers=1, widths=20, normalise_output=True).to(
            device
        )
        x = torch.randn(8, 10, device=device)
        output = mlp(x)

        # Output should be normalized
        norms = torch.norm(output, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_mlp_sota_config(self, device, mock_config):
        """Test MLP with SOTA configuration parameters."""
        # Reaction encoder: 2048 → 4096 → 512
        query_config = mock_config["model"]["query_encoder_kwargs"]
        query_mlp = MLP(**query_config).to(device)

        x = torch.randn(16, query_config["input_dim"], device=device)
        output = query_mlp(x)
        assert output.shape == (16, query_config["output_dim"])

        # Check normalization
        if query_config["normalise_output"]:
            norms = torch.norm(output, p=2, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

        # Protein encoder: 1024 → 4096 → 512
        target_config = mock_config["model"]["target_encoder_kwargs"]
        target_mlp = MLP(**target_config).to(device)

        x = torch.randn(16, target_config["input_dim"], device=device)
        output = target_mlp(x)
        assert output.shape == (16, target_config["output_dim"])

    def test_mlp_with_different_widths(self, device):
        """Test MLP with different widths for each layer."""
        widths = [64, 32, 16]
        mlp = MLP(input_dim=10, output_dim=5, widths=widths).to(device)
        x = torch.randn(4, 10, device=device)
        output = mlp(x)
        assert output.shape == (4, 5)

    def test_mlp_with_layer_norm(self, device):
        """Test MLP with layer normalization."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=2, widths=20, use_layer_norm=True).to(
            device
        )
        x = torch.randn(4, 10, device=device)
        output = mlp(x)
        assert output.shape == (4, 5)

        # Check that LayerNorm layers are in the model
        has_layer_norm = any(isinstance(layer, nn.LayerNorm) for layer in mlp.main_nn)
        assert has_layer_norm

    def test_mlp_with_dropout(self, device):
        """Test MLP with dropout."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=2, widths=20, dropout=0.5).to(device)
        x = torch.randn(4, 10, device=device)
        output = mlp(x)
        assert output.shape == (4, 5)

        # Check that Dropout layers are in the model
        has_dropout = any(isinstance(layer, nn.Dropout) for layer in mlp.main_nn)
        assert has_dropout

    def test_invalid_dropout_raises(self):
        """Dropout outside [0,1] should raise ValueError."""
        with pytest.raises(ValueError, match="dropout must be in the range"):
            MLP(input_dim=10, output_dim=5, num_layers=1, widths=20, dropout=1.5)
        with pytest.raises(ValueError, match="dropout must be in the range"):
            MLP(input_dim=10, output_dim=5, num_layers=1, widths=20, dropout=-0.1)

    def test_mlp_parameter_count(self):
        """Test parameter counting in MLP."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=1, widths=20)
        # Params: (10*20 + 20) + (20*5 + 5) = 245
        expected_params = (10 * 20 + 20) + (20 * 5 + 5)
        assert mlp.num_parameters == expected_params

    def test_invalid_widths_raises(self):
        """Zero or negative widths should raise ValueError when layers > 0."""
        with pytest.raises(ValueError, match="widths must be a positive integer"):
            MLP(input_dim=10, output_dim=5, num_layers=1, widths=0)
        with pytest.raises(ValueError, match="positive integers"):
            MLP(input_dim=10, output_dim=5, num_layers=2, widths=[16, 0])
        with pytest.raises(ValueError, match="non-empty"):
            MLP(input_dim=10, output_dim=5, num_layers=1, widths=[])

    def test_mlp_gradient_flow(self, device):
        """Test that gradients flow correctly through MLP."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=2, widths=20).to(device)
        x = torch.randn(4, 10, device=device, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Check that model parameters have gradients
        for param in mlp.parameters():
            assert param.grad is not None

    def test_mlp_no_bias(self, device):
        """Test MLP without bias terms."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=1, widths=20, bias=False).to(device)
        x = torch.randn(4, 10, device=device)
        output = mlp(x)
        assert output.shape == (4, 5)

        # Check that linear layers have no bias
        for layer in mlp.main_nn:
            if isinstance(layer, nn.Linear):
                assert layer.bias is None

    def test_num_layers_negative_raises(self):
        """Negative num_layers should raise ValueError."""
        with pytest.raises(ValueError, match="num_layers must be >= 0"):
            MLP(input_dim=10, output_dim=5, num_layers=-1, widths=20)

    def test_mlp_custom_activation(self, device):
        """Test MLP with custom activation functions."""
        mlp = MLP(
            input_dim=10,
            output_dim=5,
            num_layers=2,
            widths=20,
            activations=[nn.GELU(), nn.Tanh()],
        ).to(device)
        x = torch.randn(4, 10, device=device)
        output = mlp(x)
        assert output.shape == (4, 5)

    def test_activation_length_mismatch_raises(self):
        """Mismatch between activations and hidden layers should raise ValueError."""
        with pytest.raises(ValueError, match="Number of activations must match"):
            MLP(input_dim=10, output_dim=5, num_layers=2, widths=[16, 16], activations=[nn.ReLU()])

    def test_single_activation_instance_is_copied(self):
        """Supplying a single activation instance should result in distinct module copies per layer."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=2, widths=20, activations=nn.GELU())
        # Extract activation modules
        activs = [layer for layer in mlp.main_nn if isinstance(layer, nn.GELU)]
        assert len(activs) == 2
        assert activs[0] is not activs[1]

    def test_mlp_single_width_multiple_layers(self, device):
        """Test that single width int creates multiple layers of same width."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=3, widths=20).to(device)
        x = torch.randn(4, 10, device=device)
        output = mlp(x)
        assert output.shape == (4, 5)

        # Count linear layers (should be 4: 3 hidden + 1 output)
        linear_layers = [layer for layer in mlp.main_nn if isinstance(layer, nn.Linear)]
        assert len(linear_layers) == 4

    def test_mlp_device_placement(self, device):
        """Test that MLP works on correct device."""
        mlp = MLP(input_dim=10, output_dim=5).to(device)
        x = torch.randn(4, 10, device=device)
        output = mlp(x)

        assert output.device == x.device
        for param in mlp.parameters():
            assert param.device == x.device

    def test_mlp_batch_processing(self, device):
        """Test MLP with different batch sizes."""
        mlp = MLP(input_dim=10, output_dim=5, num_layers=1, widths=20).to(device)

        # Test with different batch sizes
        for batch_size in [1, 8, 16, 128]:
            x = torch.randn(batch_size, 10, device=device)
            output = mlp(x)
            assert output.shape == (batch_size, 5)

    def test_mlp_output_without_normalization(self, device):
        """Test that output is not normalized when normalise_output=False."""
        mlp = MLP(input_dim=10, output_dim=512, num_layers=1, widths=20, normalise_output=False).to(
            device
        )
        x = torch.randn(8, 10, device=device)
        output = mlp(x)

        # Output should NOT be normalized
        norms = torch.norm(output, p=2, dim=-1)
        # Most outputs should not have unit norm (with high probability)
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2)


class TestDualContrastiveModel:
    """Tests for DualContrastiveModel."""

    def test_initialization(self, mock_config):
        """Test basic initialization of DualContrastiveModel."""
        model_config = mock_config["model"]
        model = DualContrastiveModel(
            query_encoder_kwargs=model_config["query_encoder_kwargs"],
            target_encoder_kwargs=model_config["target_encoder_kwargs"],
        )
        assert isinstance(model.query_encoder, MLP)
        assert isinstance(model.target_encoder, MLP)

    def test_forward_pass(self, device, mock_config):
        """Test forward pass with tensor inputs."""
        model_config = mock_config["model"]
        model = DualContrastiveModel(
            query_encoder_kwargs=model_config["query_encoder_kwargs"],
            target_encoder_kwargs=model_config["target_encoder_kwargs"],
        ).to(device)

        batch_size = 16
        query_inputs = torch.randn(
            batch_size, model_config["query_encoder_kwargs"]["input_dim"], device=device
        )
        target_inputs = torch.randn(
            batch_size, model_config["target_encoder_kwargs"]["input_dim"], device=device
        )

        query_out, target_out = model(query_inputs, target_inputs)

        # Check shapes
        assert query_out.shape == (batch_size, model_config["query_encoder_kwargs"]["output_dim"])
        assert target_out.shape == (
            batch_size,
            model_config["target_encoder_kwargs"]["output_dim"],
        )

    def test_embeddings_normalized(self, device, mock_config):
        """Test that output embeddings are normalized."""
        model_config = mock_config["model"]
        model = DualContrastiveModel(
            query_encoder_kwargs=model_config["query_encoder_kwargs"],
            target_encoder_kwargs=model_config["target_encoder_kwargs"],
        ).to(device)

        batch_size = 8
        query_inputs = torch.randn(
            batch_size, model_config["query_encoder_kwargs"]["input_dim"], device=device
        )
        target_inputs = torch.randn(
            batch_size, model_config["target_encoder_kwargs"]["input_dim"], device=device
        )

        query_out, target_out = model(query_inputs, target_inputs)

        # Check normalization
        query_norms = torch.norm(query_out, p=2, dim=-1)
        target_norms = torch.norm(target_out, p=2, dim=-1)
        assert torch.allclose(query_norms, torch.ones_like(query_norms), atol=1e-6)
        assert torch.allclose(target_norms, torch.ones_like(target_norms), atol=1e-6)

    def test_sota_configuration(self, device):
        """Test SOTA model configuration from paper."""
        # SOTA: Query 2048→4096→512, Target 1024→4096→512
        model = DualContrastiveModel(
            query_encoder_kwargs={
                "input_dim": 2048,
                "output_dim": 512,
                "num_layers": 1,
                "widths": 4096,
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 1024,
                "output_dim": 512,
                "num_layers": 1,
                "widths": 4096,
                "normalise_output": True,
            },
        ).to(device)

        batch_size = 16
        query_fps = torch.randn(batch_size, 2048, device=device)  # Reaction fingerprints
        target_embs = torch.randn(batch_size, 1024, device=device)  # Protein T5 embeddings

        query_out, target_out = model(query_fps, target_embs)

        assert query_out.shape == (batch_size, 512)
        assert target_out.shape == (batch_size, 512)

        # Check normalization
        query_norms = torch.norm(query_out, p=2, dim=-1)
        target_norms = torch.norm(target_out, p=2, dim=-1)
        assert torch.allclose(query_norms, torch.ones_like(query_norms), atol=1e-6)
        assert torch.allclose(target_norms, torch.ones_like(target_norms), atol=1e-6)

    def test_dimension_mismatch_error(self, device):
        """Test that dimension mismatch raises error."""
        model = DualContrastiveModel(
            query_encoder_kwargs={
                "input_dim": 100,
                "output_dim": 512,
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 100,
                "output_dim": 256,  # Different output dim
                "normalise_output": True,
            },
        ).to(device)

        query_inputs = torch.randn(8, 100, device=device)
        target_inputs = torch.randn(8, 100, device=device)

        with pytest.raises(ValueError, match="output shape mismatch"):
            model(query_inputs, target_inputs)

    def test_enforce_normalization(self):
        """Test that enforce_normalisation validates NormalizeLayer presence."""
        # Should raise error when normalise_output=False
        with pytest.raises(ValueError, match="must have a NormalizeLayer"):
            DualContrastiveModel(
                query_encoder_kwargs={
                    "input_dim": 100,
                    "output_dim": 512,
                    "normalise_output": False,  # No normalization
                },
                target_encoder_kwargs={
                    "input_dim": 100,
                    "output_dim": 512,
                    "normalise_output": True,
                },
                enforce_normalisation=True,
            )

    def test_disable_normalization_enforcement(self, device):
        """Test that normalization enforcement can be disabled."""
        # Should work fine when enforcement is disabled
        model = DualContrastiveModel(
            query_encoder_kwargs={
                "input_dim": 100,
                "output_dim": 512,
                "normalise_output": False,
            },
            target_encoder_kwargs={
                "input_dim": 100,
                "output_dim": 512,
                "normalise_output": False,
            },
            enforce_normalisation=False,
        ).to(device)

        query_inputs = torch.randn(8, 100, device=device)
        target_inputs = torch.randn(8, 100, device=device)
        query_out, target_out = model(query_inputs, target_inputs)

        assert query_out.shape == (8, 512)
        assert target_out.shape == (8, 512)

    def test_gradient_flow(self, device, mock_config):
        """Test that gradients flow through both encoders."""
        model_config = mock_config["model"]
        model = DualContrastiveModel(
            query_encoder_kwargs=model_config["query_encoder_kwargs"],
            target_encoder_kwargs=model_config["target_encoder_kwargs"],
        ).to(device)

        query_inputs = torch.randn(
            8, model_config["query_encoder_kwargs"]["input_dim"], device=device, requires_grad=True
        )
        target_inputs = torch.randn(
            8,
            model_config["target_encoder_kwargs"]["input_dim"],
            device=device,
            requires_grad=True,
        )

        query_out, target_out = model(query_inputs, target_inputs)
        loss = query_out.sum() + target_out.sum()
        loss.backward()

        # Check gradients exist
        assert query_inputs.grad is not None
        assert target_inputs.grad is not None

        # Check model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None

    def test_dict_inputs(self, device):
        """Test forward pass with dictionary inputs."""
        model = DualContrastiveModel(
            query_encoder_kwargs={"input_dim": 100, "output_dim": 512, "normalise_output": True},
            target_encoder_kwargs={"input_dim": 100, "output_dim": 512, "normalise_output": True},
        ).to(device)

        # Note: MLPs don't actually support dict inputs, but the code path exists
        # for compatibility with other encoder types. Test with tensors instead.
        query_inputs = torch.randn(8, 100, device=device)
        target_inputs = torch.randn(8, 100, device=device)

        query_out, target_out = model(query_inputs, target_inputs)
        assert query_out.shape == (8, 512)
        assert target_out.shape == (8, 512)

    def test_different_batch_sizes(self, device):
        """Test model with various batch sizes."""
        model = DualContrastiveModel(
            query_encoder_kwargs={"input_dim": 100, "output_dim": 512, "normalise_output": True},
            target_encoder_kwargs={"input_dim": 100, "output_dim": 512, "normalise_output": True},
        ).to(device)

        for batch_size in [1, 8, 16, 128]:
            query_inputs = torch.randn(batch_size, 100, device=device)
            target_inputs = torch.randn(batch_size, 100, device=device)

            query_out, target_out = model(query_inputs, target_inputs)
            assert query_out.shape == (batch_size, 512)
            assert target_out.shape == (batch_size, 512)

    def test_parameter_count(self):
        """Test total parameter counting in dual model."""
        model = DualContrastiveModel(
            query_encoder_kwargs={
                "input_dim": 100,
                "output_dim": 50,
                "num_layers": 1,
                "widths": 200,
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 150,
                "output_dim": 50,
                "num_layers": 1,
                "widths": 200,
                "normalise_output": True,
            },
        )

        # Query encoder: (100*200 + 200) + (200*50 + 50) = 20200 + 10050 = 30250
        # Target encoder: (150*200 + 200) + (200*50 + 50) = 30200 + 10050 = 40250
        # Total: 70500
        expected_params = 30250 + 40250
        assert model.num_parameters == expected_params


class TestProteinPooledDualModelHybridEnzymeInput:
    """Tests for hybrid raw ProT5 plus hyperbolic enzyme input wiring."""

    def test_hybrid_mode_requires_hyperbolic_checkpoint(self):
        with pytest.raises(ValueError, match="requires hyperbolic_checkpoint_path"):
            ProteinPooledDualModel(
                query_encoder_kwargs={
                    "input_dim": 3,
                    "output_dim": 2,
                    "normalise_output": True,
                },
                target_encoder_kwargs={
                    "input_dim": 6,
                    "output_dim": 2,
                    "normalise_output": True,
                },
                residue_dim=4,
                pooling="mean",
                enzyme_input_mode="raw_sleec_hyperbolic_concat",
            )

    def test_hybrid_target_input_concatenates_raw_pool_and_tangent(self, device):
        class FakeProjector(nn.Module):
            def forward(self, pooled):
                z_hyp = torch.full(
                    (pooled.shape[0], 3),
                    2.0,
                    dtype=pooled.dtype,
                    device=pooled.device,
                )
                z_tangent = pooled[:, :2] + 10.0
                return z_hyp, z_tangent

        class CaptureEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_input = None

            def forward(self, target_input):
                self.last_input = target_input.detach()
                return F.normalize(target_input[:, :2], dim=-1)

        model = ProteinPooledDualModel(
            query_encoder_kwargs={
                "input_dim": 3,
                "output_dim": 2,
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 4,
                "output_dim": 2,
                "normalise_output": True,
            },
            residue_dim=4,
            pooling="mean",
        ).to(device)
        model.enzyme_input_mode = "raw_sleec_hyperbolic_concat"
        model.hyperbolic_use_tangent = True
        model.hyperbolic_projector = FakeProjector().to(device)
        capture_encoder = CaptureEncoder().to(device)
        model.target_encoder = capture_encoder

        residues = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
                [[2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            device=device,
        )
        padding_mask = torch.tensor(
            [[False, False], [False, True]],
            dtype=torch.bool,
            device=device,
        )

        output = model.encode_targets(residues, residue_padding_mask=padding_mask)

        expected_raw = torch.tensor(
            [[2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0]],
            device=device,
        )
        expected_tangent = expected_raw[:, :2] + 10.0
        expected_input = torch.cat([expected_raw, expected_tangent], dim=-1)

        assert output.shape == (2, 2)
        assert capture_encoder.last_input is not None
        assert capture_encoder.last_input.shape == (2, 6)
        assert torch.allclose(capture_encoder.last_input, expected_input)

    def test_gated_hybrid_receives_raw_pool_and_tangent(self, device):
        class FixedPool(nn.Module):
            def forward(self, residues, attention_mask=None, return_attention=False):
                pooled = residues[:, 0, :] + 100.0
                if return_attention:
                    weights = torch.zeros(residues.shape[:2], device=residues.device)
                    weights[:, 0] = 1.0
                    return pooled, weights
                return pooled

        class RawMeanPool(nn.Module):
            def forward(self, residues, attention_mask=None):
                valid = attention_mask.to(dtype=residues.dtype, device=residues.device)
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                return (residues * valid.unsqueeze(-1)).sum(dim=1) / denom

        class FakeProjector(nn.Module):
            def forward(self, pooled):
                z_hyp = torch.zeros(
                    (pooled.shape[0], 3),
                    dtype=pooled.dtype,
                    device=pooled.device,
                )
                z_tangent = pooled[:, :3] - 50.0
                return z_hyp, z_tangent

        class CaptureFusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_raw_mean = None
                self.last_pooled = None
                self.last_tangent = None

            def forward(
                self,
                raw_mean,
                pooled,
                hyperbolic_tangent,
                capability_vector=None,
                capability_mask=None,
                return_gates=False,
            ):
                self.last_raw_mean = raw_mean.detach()
                self.last_pooled = pooled.detach()
                self.last_tangent = hyperbolic_tangent.detach()
                fused = raw_mean + pooled
                if return_gates:
                    gates = torch.tensor(
                        [[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]],
                        dtype=raw_mean.dtype,
                        device=raw_mean.device,
                    )
                    return fused, gates
                return fused

        class CaptureEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_input = None

            def forward(self, target_input):
                self.last_input = target_input.detach()
                return F.normalize(target_input[:, :2], dim=-1)

        model = ProteinPooledDualModel(
            query_encoder_kwargs={
                "input_dim": 3,
                "output_dim": 2,
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 4,
                "output_dim": 2,
                "normalise_output": True,
            },
            residue_dim=4,
            pooling="mean",
        ).to(device)
        model.pooling = FixedPool().to(device)
        model.enzyme_input_mode = "raw_mean_sleec_hyperbolic_gated"
        model.hyperbolic_use_tangent = True
        model.hyperbolic_projector = FakeProjector().to(device)
        model.raw_mean_pooling = RawMeanPool().to(device)
        capture_fusion = CaptureFusion().to(device)
        model.enzyme_feature_fusion = capture_fusion
        capture_encoder = CaptureEncoder().to(device)
        model.target_encoder = capture_encoder

        residues = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
                [[2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            device=device,
        )
        padding_mask = torch.tensor(
            [[False, False], [False, True]],
            dtype=torch.bool,
            device=device,
        )

        output, details = model.encode_targets(
            residues,
            residue_padding_mask=padding_mask,
            return_pooling_details=True,
        )

        expected_raw = torch.tensor(
            [[2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0]],
            device=device,
        )
        expected_pooled = torch.tensor(
            [[101.0, 102.0, 103.0, 104.0], [102.0, 104.0, 106.0, 108.0]],
            device=device,
        )
        expected_tangent = expected_pooled[:, :3] - 50.0

        assert output.shape == (2, 2)
        assert torch.allclose(capture_fusion.last_raw_mean, expected_raw)
        assert torch.allclose(capture_fusion.last_pooled, expected_pooled)
        assert torch.allclose(capture_fusion.last_tangent, expected_tangent)
        assert capture_encoder.last_input is not None
        assert torch.allclose(capture_encoder.last_input, expected_raw + expected_pooled)
        assert torch.allclose(
            details["enzyme_fusion_gate_raw_mean"],
            torch.tensor([0.2, 0.1], device=device),
        )
        assert torch.allclose(
            details["enzyme_fusion_gate_pooled"],
            torch.tensor([0.3, 0.6], device=device),
        )
        assert torch.allclose(
            details["enzyme_fusion_gate_hyperbolic"],
            torch.tensor([0.5, 0.3], device=device),
        )

    def test_tiger_text_fusion_masks_missing_text(self, device):
        fusion = TigerTextGatedFusion(
            sequence_dim=4,
            text_dim=6,
            output_dim=4,
            fusion_dim=8,
            num_heads=2,
            dropout=0.0,
        ).to(device)
        sequence = torch.randn(2, 4, device=device)
        text = torch.randn(2, 6, device=device)
        output, details = fusion(
            sequence,
            text,
            text_mask=torch.tensor([True, False], device=device),
            return_details=True,
        )
        assert output.shape == (2, 4)
        assert details["text_fusion_alpha"].shape == (2, 8)
        assert details["text_fusion_has_text"].tolist() == [1.0, 0.0]
        assert torch.allclose(output[1], sequence[1])

    def test_text_gated_hybrid_receives_text_vectors(self, device):
        class FixedPool(nn.Module):
            def forward(self, residues, attention_mask=None, return_attention=False):
                pooled = residues[:, 0, :] + 10.0
                if return_attention:
                    weights = torch.zeros(residues.shape[:2], device=residues.device)
                    weights[:, 0] = 1.0
                    return pooled, weights
                return pooled

        class RawMeanPool(nn.Module):
            def forward(self, residues, attention_mask=None):
                valid = attention_mask.to(dtype=residues.dtype, device=residues.device)
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                return (residues * valid.unsqueeze(-1)).sum(dim=1) / denom

        class FakeProjector(nn.Module):
            def forward(self, pooled):
                z_hyp = torch.zeros(
                    (pooled.shape[0], 3),
                    dtype=pooled.dtype,
                    device=pooled.device,
                )
                z_tangent = pooled[:, :3]
                return z_hyp, z_tangent

        class CaptureFusion(nn.Module):
            def forward(
                self,
                raw_mean,
                pooled,
                hyperbolic_tangent,
                capability_vector=None,
                capability_mask=None,
                return_gates=False,
            ):
                fused = raw_mean + pooled
                if return_gates:
                    gates = torch.full(
                        (raw_mean.shape[0], 3),
                        1.0 / 3.0,
                        dtype=raw_mean.dtype,
                        device=raw_mean.device,
                    )
                    return fused, gates
                return fused

        class CaptureTextFusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_sequence = None
                self.last_text = None
                self.last_mask = None

            def forward(self, sequence_feature, text_vector, text_mask=None, return_details=False):
                self.last_sequence = sequence_feature.detach()
                self.last_text = text_vector.detach()
                self.last_mask = None if text_mask is None else text_mask.detach()
                text_delta = torch.zeros_like(sequence_feature)
                text_delta[:, : text_vector.shape[1]] = text_vector[:, : sequence_feature.shape[1]]
                fused = torch.where(
                    text_mask.unsqueeze(-1),
                    sequence_feature + text_delta,
                    sequence_feature,
                )
                if return_details:
                    return fused, {
                        "text_fusion_alpha": torch.ones_like(sequence_feature),
                        "text_fusion_has_text": text_mask.to(dtype=sequence_feature.dtype),
                    }
                return fused

        class CaptureEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_input = None

            def forward(self, target_input):
                self.last_input = target_input.detach()
                return F.normalize(target_input[:, :2], dim=-1)

        model = ProteinPooledDualModel(
            query_encoder_kwargs={
                "input_dim": 3,
                "output_dim": 2,
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 4,
                "output_dim": 2,
                "normalise_output": True,
            },
            residue_dim=4,
            pooling="mean",
        ).to(device)
        model.pooling = FixedPool().to(device)
        model.enzyme_input_mode = "raw_mean_sleec_hyperbolic_text_gated"
        model.hyperbolic_use_tangent = True
        model.hyperbolic_projector = FakeProjector().to(device)
        model.raw_mean_pooling = RawMeanPool().to(device)
        model.enzyme_feature_fusion = CaptureFusion().to(device)
        capture_text_fusion = CaptureTextFusion().to(device)
        model.text_feature_fusion = capture_text_fusion
        capture_encoder = CaptureEncoder().to(device)
        model.target_encoder = capture_encoder

        residues = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
                [[2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            device=device,
        )
        padding_mask = torch.tensor(
            [[False, False], [False, True]],
            dtype=torch.bool,
            device=device,
        )
        text_vectors = torch.tensor(
            [[0.5, 1.0, 1.5, 2.0], [9.0, 9.0, 9.0, 9.0]],
            device=device,
        )
        text_mask = torch.tensor([True, False], dtype=torch.bool, device=device)

        output, details = model.encode_targets(
            residues,
            residue_padding_mask=padding_mask,
            text_vectors=text_vectors,
            text_mask=text_mask,
            return_pooling_details=True,
        )

        expected_raw = torch.tensor(
            [[2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0]],
            device=device,
        )
        expected_pooled = torch.tensor(
            [[11.0, 12.0, 13.0, 14.0], [12.0, 14.0, 16.0, 18.0]],
            device=device,
        )
        expected_sequence = expected_raw + expected_pooled
        expected_target_input = expected_sequence.clone()
        expected_target_input[0] = expected_target_input[0] + text_vectors[0]

        assert output.shape == (2, 2)
        assert torch.allclose(capture_text_fusion.last_sequence, expected_sequence)
        assert torch.allclose(capture_text_fusion.last_text, text_vectors)
        assert torch.equal(capture_text_fusion.last_mask, text_mask)
        assert torch.allclose(capture_encoder.last_input, expected_target_input)
        assert "text_fusion_alpha" in details
        assert torch.allclose(details["text_fusion_has_text"], text_mask.float())

    def test_biofp_split_mode_returns_embedding_and_auxiliary_logits(self, device):
        model = ProteinPooledDualModel(
            query_encoder_kwargs={
                "input_dim": 3,
                "output_dim": 6,
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 6,
                "output_dim": 6,
                "normalise_output": True,
            },
            residue_dim=4,
            pooling="mean",
            enzyme_input_mode="raw_mean_sleec_biofp_split",
            biofp_center_dim=2,
            biofp_cofactor_dim=1,
            biofp_transition_dim=3,
            biofp_seq_dim=4,
            biofp_dim=2,
            biofp_hidden_dim=8,
            biofp_seq_weight=0.75,
            biofp_dropout=0.0,
        ).to(device)

        residues = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]],
                [[2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            device=device,
        )
        padding_mask = torch.tensor(
            [[False, False], [False, True]],
            dtype=torch.bool,
            device=device,
        )

        output, details = model.encode_targets(
            residues,
            residue_padding_mask=padding_mask,
            return_pooling_details=True,
        )

        assert output.shape == (2, 6)
        assert torch.allclose(output.norm(dim=-1), torch.ones(2, device=device), atol=1e-5)
        assert details["biofp_logits_center"].shape == (2, 2)
        assert details["biofp_logits_cofactor"].shape == (2, 1)
        assert details["biofp_logits_transition"].shape == (2, 3)
        assert details["biofp_sequence_gate_raw_mean"].shape == (2,)
        assert details["biofp_attention_entropy_center"].shape == (2,)
        assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())

    def test_biological_factorized_mode_is_sequence_only_and_fixed_layout(self, device):
        model = ProteinPooledDualModel(
            query_encoder_kwargs={
                "input_dim": 3,
                "output_dim": 10,
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 10,
                "output_dim": 10,
                "normalise_output": True,
            },
            residue_dim=4,
            pooling="sleec_guided_attention",
            sleec_scorer_hidden_dim=4,
            enzyme_input_mode="raw_mean_sleec_biological_factorized",
            hyperbolic_hyp_dim=3,
            enzyme_block_dims={
                "core": 3,
                "site": 2,
                "mechanism": 2,
                "cofactor": 1,
                "ec": 2,
            },
            enzyme_block_weights={
                "core": 0.55,
                "site": 0.20,
                "mechanism": 0.12,
                "cofactor": 0.08,
                "ec": 0.05,
            },
            biofp_family_dims={"mechanism": 2, "cofactor": 1},
            biofp_hidden_dim=8,
            biofp_dropout=0.0,
        ).to(device)
        residues = torch.randn(2, 4, 4, device=device)
        padding_mask = torch.tensor(
            [[False, False, False, True], [False, False, True, True]],
            device=device,
        )

        model.eval()
        output, details = model.encode_targets(
            residues,
            residue_padding_mask=padding_mask,
            return_pooling_details=True,
        )
        with torch.no_grad():
            output_without_diagnostics = model.encode_targets(
                residues,
                residue_padding_mask=padding_mask,
            )

        assert output.shape == (2, 10)
        assert torch.allclose(output.norm(dim=-1), torch.ones(2, device=device), atol=1e-5)
        assert torch.allclose(output, output_without_diagnostics, atol=1e-6)
        assert details["biofp_logits_mechanism"].shape == (2, 2)
        assert details["biofp_logits_cofactor"].shape == (2, 1)
        details["biofp_logits_mechanism"].sum().backward()
        mechanism_projection = model.biological_factorized_encoder.family_projections[
            "mechanism"
        ][1]
        assert mechanism_projection.weight.grad is not None
        for name in ("core", "site", "mechanism", "cofactor", "ec"):
            assert f"enzyme_block_norm_{name}" in details
        assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())

    def test_biological_factorized_supports_reduced_learned_block_layout(self, device):
        model = ProteinPooledDualModel(
            query_encoder_kwargs={
                "input_dim": 3,
                "output_dim": 8,
                "normalise_output": True,
            },
            target_encoder_kwargs={
                "input_dim": 8,
                "output_dim": 8,
                "normalise_output": True,
            },
            residue_dim=4,
            pooling="sleec_guided_attention",
            sleec_scorer_hidden_dim=4,
            enzyme_input_mode="raw_mean_sleec_biological_factorized",
            enzyme_block_dims={"core": 6, "site": 2},
            enzyme_block_weights={"core": 0.75, "site": 0.25},
            biofp_family_dims={},
            biofp_hidden_dim=8,
            biofp_dropout=0.0,
            enzyme_block_learned_weights=True,
        ).to(device)
        residues = torch.randn(3, 5, 4, device=device)
        padding_mask = torch.zeros(3, 5, dtype=torch.bool, device=device)

        output, details = model.encode_targets(
            residues,
            residue_padding_mask=padding_mask,
            return_pooling_details=True,
        )

        assert output.shape == (3, 8)
        assert model.hyperbolic_projector is None
        assert torch.allclose(output.norm(dim=-1), torch.ones(3, device=device), atol=1e-5)
        assert details["enzyme_block_weight_core"][0].item() == pytest.approx(0.75)
        assert details["enzyme_block_weight_site"][0].item() == pytest.approx(0.25)
        assert details["enzyme_block_weight_kl"].abs().max().item() < 1e-6
        output.sum().backward()
        assert model.biological_factorized_encoder.block_weight_logits.grad is not None
