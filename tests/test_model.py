"""Tests for SimpleDLRM model and DLRMConfig."""

import pytest
import torch
import torch.nn as nn

from training.models.simple_dlrm import DLRMConfig, SimpleDLRM, create_model_from_mappings


# ---------------------------------------------------------------------------
# DLRMConfig tests
# ---------------------------------------------------------------------------


class TestDLRMConfig:
    """Tests for the DLRMConfig dataclass."""

    def test_defaults(self) -> None:
        """DLRMConfig should have sensible defaults for all fields."""
        cfg = DLRMConfig()
        assert cfg.num_users == 1000
        assert cfg.num_items == 1000
        assert cfg.embedding_dim == 16
        assert cfg.dense_feature_dim == 6
        assert cfg.bottom_mlp_dims == (64, 32, 16)
        assert cfg.top_mlp_dims == (64, 32, 1)
        assert cfg.dropout_rate == 0.1

    def test_custom_values(self) -> None:
        """DLRMConfig should accept custom values for all fields."""
        cfg = DLRMConfig(
            num_users=500,
            num_items=2000,
            embedding_dim=32,
            dense_feature_dim=10,
            bottom_mlp_dims=(128, 64),
            top_mlp_dims=(128, 1),
            dropout_rate=0.2,
        )
        assert cfg.num_users == 500
        assert cfg.num_items == 2000
        assert cfg.embedding_dim == 32
        assert cfg.dense_feature_dim == 10

    def test_num_sparse_features_property(self) -> None:
        """num_sparse_features should always be 2 (user + item)."""
        cfg = DLRMConfig()
        assert cfg.num_sparse_features == 2

    def test_interaction_output_dim_property(self) -> None:
        """interaction_output_dim = n*(n+1)/2 where n = num_sparse + 1."""
        cfg = DLRMConfig()
        # n = 2 sparse + 1 dense = 3, so 3*(3+1)//2 = 6
        assert cfg.interaction_output_dim == 6


# ---------------------------------------------------------------------------
# SimpleDLRM tests
# ---------------------------------------------------------------------------


class TestSimpleDLRM:
    """Tests for the SimpleDLRM model."""

    def test_forward_pass_shape_batch4(self, model_config: DLRMConfig) -> None:
        """Forward pass with batch=4 should return shape (4,)."""
        model = SimpleDLRM(config=model_config)
        model.eval()

        batch_size = 4
        user_ids = torch.randint(0, model_config.num_users, (batch_size,))
        item_ids = torch.randint(0, model_config.num_items, (batch_size,))
        dense = torch.randn(batch_size, model_config.dense_feature_dim)

        with torch.no_grad():
            output = model(user_ids, item_ids, dense)

        assert output.shape == (batch_size,)

    def test_output_range_sigmoid(self, trained_model: SimpleDLRM) -> None:
        """All outputs should be in [0, 1] due to sigmoid activation."""
        cfg = trained_model.config
        batch_size = 32

        user_ids = torch.randint(0, cfg.num_users, (batch_size,))
        item_ids = torch.randint(0, cfg.num_items, (batch_size,))
        dense = torch.randn(batch_size, cfg.dense_feature_dim)

        with torch.no_grad():
            output = trained_model(user_ids, item_ids, dense)

        assert (output >= 0.0).all(), "Some outputs below 0"
        assert (output <= 1.0).all(), "Some outputs above 1"

    def test_gradient_flow(self, model_config: DLRMConfig) -> None:
        """loss.backward() should execute without error and populate gradients."""
        model = SimpleDLRM(config=model_config)
        model.train()

        batch_size = 4
        user_ids = torch.randint(0, model_config.num_users, (batch_size,))
        item_ids = torch.randint(0, model_config.num_items, (batch_size,))
        dense = torch.randn(batch_size, model_config.dense_feature_dim)
        targets = torch.rand(batch_size)

        output = model(user_ids, item_ids, dense)
        loss = nn.BCELoss()(output, targets)
        loss.backward()

        # At least one parameter should have a non-None gradient
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, "No gradients were computed"

    def test_parameter_count_reasonable(self, model_config: DLRMConfig) -> None:
        """Total parameter count should be positive and reasonable."""
        model = SimpleDLRM(config=model_config)

        total = model.count_parameters()
        assert total > 0, "Parameter count must be positive"
        # With small config, should be at most a few hundred thousand
        assert total < 10_000_000, "Parameter count suspiciously large for test config"

    def test_get_parameter_count_breakdown(self, model_config: DLRMConfig) -> None:
        """get_parameter_count should return a dict with expected component keys."""
        model = SimpleDLRM(config=model_config)
        counts = model.get_parameter_count()

        expected_keys = {"user_embedding", "item_embedding", "bottom_mlp", "top_mlp", "total"}
        assert set(counts.keys()) == expected_keys

        # Component counts should sum to total
        component_sum = (
            counts["user_embedding"]
            + counts["item_embedding"]
            + counts["bottom_mlp"]
            + counts["top_mlp"]
        )
        assert component_sum == counts["total"]

    def test_single_sample_batch1(self, model_config: DLRMConfig) -> None:
        """Model should handle a single sample (batch_size=1) correctly."""
        model = SimpleDLRM(config=model_config)
        model.eval()

        user_ids = torch.tensor([0])
        item_ids = torch.tensor([0])
        dense = torch.randn(1, model_config.dense_feature_dim)

        with torch.no_grad():
            output = model(user_ids, item_ids, dense)

        assert output.shape == (1,)
        assert 0.0 <= output.item() <= 1.0

    def test_init_with_kwargs(self) -> None:
        """SimpleDLRM should accept keyword arguments instead of a config object."""
        model = SimpleDLRM(num_users=20, num_items=30, embedding_dim=8)
        assert model.config.num_users == 20
        assert model.config.num_items == 30
        assert model.config.embedding_dim == 8


# ---------------------------------------------------------------------------
# create_model_from_mappings tests
# ---------------------------------------------------------------------------


class TestCreateModelFromMappings:
    """Tests for the create_model_from_mappings factory function."""

    def test_produces_correct_embedding_sizes(self) -> None:
        """Embedding tables should match the mapping dictionary sizes."""
        user_map = {f"u{i}": i for i in range(25)}
        item_map = {f"i{i}": i for i in range(75)}

        model = create_model_from_mappings(user_map, item_map, embedding_dim=8)

        assert model.config.num_users == 25
        assert model.config.num_items == 75
        assert model.config.embedding_dim == 8
        assert model.user_embedding.num_embeddings == 25
        assert model.item_embedding.num_embeddings == 75

    def test_default_dense_feature_dim(self) -> None:
        """Default dense_feature_dim should be 6."""
        user_map = {"a": 0}
        item_map = {"b": 0}
        model = create_model_from_mappings(user_map, item_map)
        assert model.config.dense_feature_dim == 6

    def test_empty_mappings_floor_to_one(self) -> None:
        """Empty mappings should result in at least 1 embedding row (max(len, 1))."""
        model = create_model_from_mappings({}, {})
        assert model.config.num_users == 1
        assert model.config.num_items == 1

    def test_returns_simple_dlrm_instance(self) -> None:
        """Factory should return a SimpleDLRM instance."""
        model = create_model_from_mappings({"u": 0}, {"i": 0})
        assert isinstance(model, SimpleDLRM)
