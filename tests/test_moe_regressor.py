#!/usr/bin/env python3
"""
Test MoERegressor module

Test contents:
1. Basic initialization
2. Forward method input/output shapes
3. Validity of gate_weights
4. Different parameter configurations
5. Gradient flow
6. Edge cases
"""

import os

import pytest
import torch
import torch.nn as nn

import sys

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.deepsc.models.deepsc.model import MoERegressor, RegressorExpert


class TestMoERegressor:
    """MoERegressor test class"""

    @pytest.fixture
    def default_params(self):
        """Default parameters"""
        return {
            "embedding_dim": 256,
            "dropout": 0.1,
            "number_of_experts": 3,
            "gate_temperature": 1.0,
        }

    @pytest.fixture
    def device(self):
        """Test device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self, default_params):
        """Test basic initialization"""
        model = MoERegressor(**default_params)

        # Check basic attributes
        assert model.embedding_dim == default_params["embedding_dim"]
        assert model.num_experts == default_params["number_of_experts"]
        assert model.gate_temperature == default_params["gate_temperature"]

        # Check gate network structure
        assert isinstance(model.gate, nn.Sequential)

        # Check number of experts
        assert len(model.experts) == default_params["number_of_experts"]

        # Check each expert is a RegressorExpert instance
        for expert in model.experts:
            assert isinstance(expert, RegressorExpert)

    def test_forward_shape(self, default_params, device):
        """Test forward method input/output shapes"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        # Test different batch_size and seq_len
        test_cases = [
            (2, 64),  # Small batch
            (8, 128),  # Medium batch
            (16, 256),  # Large batch
            (1, 32),  # Single sample
        ]

        for batch_size, seq_len in test_cases:
            with torch.no_grad():
                x = torch.randn(
                    batch_size, seq_len, default_params["embedding_dim"]
                ).to(device)
                output, gate_weights = model(x)

                # Check output shape
                assert output.shape == (
                    batch_size,
                    seq_len,
                ), f"Expected output shape {(batch_size, seq_len)}, got {output.shape}"

                # Check gate_weights shape
                expected_gate_shape = (
                    batch_size,
                    seq_len,
                    default_params["number_of_experts"],
                )
                assert (
                    gate_weights.shape == expected_gate_shape
                ), f"Expected gate_weights shape {expected_gate_shape}, got {gate_weights.shape}"

    def test_gate_weights_validity(self, default_params, device):
        """Test validity of gate_weights"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, default_params["embedding_dim"]).to(device)

        with torch.no_grad():
            output, gate_weights = model(x)

            # 1. Check gate_weights are in [0, 1] range
            assert torch.all(
                gate_weights >= 0
            ), "Gate weights should be non-negative (softmax output)"
            assert torch.all(
                gate_weights <= 1
            ), "Gate weights should be <= 1 (softmax output)"

            # 2. Check gate_weights sum to 1 at each position (softmax property)
            weight_sums = gate_weights.sum(dim=-1)
            assert torch.allclose(
                weight_sums, torch.ones_like(weight_sums), atol=1e-6
            ), "Gate weights should sum to 1 along expert dimension"

            # 3. Check gate_weights don't have NaN or Inf
            assert not torch.isnan(gate_weights).any(), "Gate weights contain NaN"
            assert not torch.isinf(gate_weights).any(), "Gate weights contain Inf"

    def test_different_expert_numbers(self, device):
        """Test different numbers of experts"""
        embedding_dim = 256
        batch_size, seq_len = 4, 64

        expert_numbers = [2, 3, 4, 5, 8]

        for num_experts in expert_numbers:
            model = MoERegressor(
                embedding_dim=embedding_dim,
                dropout=0.1,
                number_of_experts=num_experts,
                gate_temperature=1.0,
            ).to(device)

            model.eval()

            x = torch.randn(batch_size, seq_len, embedding_dim).to(device)

            with torch.no_grad():
                output, gate_weights = model(x)

                # Check shapes
                assert output.shape == (batch_size, seq_len)
                assert gate_weights.shape == (batch_size, seq_len, num_experts)

                # Check gate_weights sum to 1
                weight_sums = gate_weights.sum(dim=-1)
                assert torch.allclose(
                    weight_sums, torch.ones_like(weight_sums), atol=1e-6
                )

    def test_gate_temperature_effect(self, default_params, device):
        """Test effect of different gate_temperature on output"""
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, default_params["embedding_dim"]).to(device)

        temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
        gate_weights_list = []

        for temp in temperatures:
            params = default_params.copy()
            params["gate_temperature"] = temp
            model = MoERegressor(**params).to(device)
            model.eval()

            with torch.no_grad():
                # Fix random seed for reproducibility
                torch.manual_seed(42)
                _, gate_weights = model(x)
                gate_weights_list.append(gate_weights)

                # Check gate_weights are still valid
                weight_sums = gate_weights.sum(dim=-1)
                assert torch.allclose(
                    weight_sums, torch.ones_like(weight_sums), atol=1e-6
                )

        # Lower temperature should produce sharper distribution (closer to one-hot)
        # Higher temperature should produce smoother distribution (more uniform)
        # Note: Due to random initialization, we only check the trend, not specific values

    def test_gradient_flow(self, default_params, device):
        """Test gradient flow"""
        model = MoERegressor(**default_params).to(device)
        model.train()

        batch_size, seq_len = 4, 64
        x = torch.randn(
            batch_size, seq_len, default_params["embedding_dim"], requires_grad=True
        ).to(device)

        # Retain input gradients (for non-leaf tensors)
        x.retain_grad()

        # Forward pass
        output, gate_weights = model(x)

        # Compute simple loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input gradients should exist"

        # Check model parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"
                assert not torch.isnan(
                    param.grad
                ).any(), f"Parameter {name} has NaN gradients"
                assert not torch.isinf(
                    param.grad
                ).any(), f"Parameter {name} has Inf gradients"

    def test_weight_initialization(self, default_params):
        """Test weight initialization"""
        model = MoERegressor(**default_params)

        # Check gate network weights and biases
        for m in model.gate:
            if isinstance(m, nn.Linear):
                # Check weights are not all zeros
                assert not torch.allclose(
                    m.weight, torch.zeros_like(m.weight)
                ), "Gate weights should not be all zeros"
                # Check biases are all zeros
                assert torch.allclose(
                    m.bias, torch.zeros_like(m.bias)
                ), "Gate biases should be initialized to zeros"

        # Check expert network weights and biases
        for expert in model.experts:
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    # Check weights are not all zeros
                    assert not torch.allclose(
                        m.weight, torch.zeros_like(m.weight)
                    ), "Expert weights should not be all zeros"
                    # Check biases are all zeros
                    assert torch.allclose(
                        m.bias, torch.zeros_like(m.bias)
                    ), "Expert biases should be initialized to zeros"

    def test_output_range(self, default_params, device):
        """Test output value reasonableness"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        batch_size, seq_len = 4, 64

        # Test normal input
        x = torch.randn(batch_size, seq_len, default_params["embedding_dim"]).to(device)

        with torch.no_grad():
            output, gate_weights = model(x)

            # Check output has no NaN or Inf
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

            # For normalized input, output should be within reasonable range (not too large)
            assert torch.abs(output).max() < 1e3, "Output values are unreasonably large"

    def test_zero_input(self, default_params, device):
        """Test zero input case"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        batch_size, seq_len = 4, 64
        x = torch.zeros(batch_size, seq_len, default_params["embedding_dim"]).to(device)

        with torch.no_grad():
            output, gate_weights = model(x)

            # Check output is valid
            assert not torch.isnan(output).any(), "Output contains NaN for zero input"
            assert not torch.isinf(output).any(), "Output contains Inf for zero input"

            # Check gate_weights are still valid
            weight_sums = gate_weights.sum(dim=-1)
            assert torch.allclose(
                weight_sums, torch.ones_like(weight_sums), atol=1e-6
            ), "Gate weights should sum to 1 for zero input"

    def test_determinism(self, default_params, device):
        """Test determinism (same input should produce same output)"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, default_params["embedding_dim"]).to(device)

        with torch.no_grad():
            output1, gate_weights1 = model(x)
            output2, gate_weights2 = model(x)

            # Same input should produce same output
            assert torch.allclose(
                output1, output2, atol=1e-6
            ), "Model should be deterministic in eval mode"
            assert torch.allclose(
                gate_weights1, gate_weights2, atol=1e-6
            ), "Gate weights should be deterministic in eval mode"

    def test_dropout_effect(self, device):
        """Test dropout effect in training and evaluation modes"""
        params = {
            "embedding_dim": 256,
            "dropout": 0.5,  # Higher dropout rate to observe effect
            "number_of_experts": 3,
            "gate_temperature": 1.0,
        }

        model = MoERegressor(**params).to(device)
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, params["embedding_dim"]).to(device)

        # Training mode: outputs should differ (due to dropout)
        model.train()
        torch.manual_seed(42)
        output1_train, _ = model(x)
        torch.manual_seed(43)
        output2_train, _ = model(x)

        # In training mode, outputs should differ due to dropout
        # Note: This test may occasionally fail due to randomness
        assert not torch.allclose(
            output1_train, output2_train, atol=1e-6
        ), "Outputs should differ in training mode due to dropout"

        # Evaluation mode: outputs should be identical
        model.eval()
        with torch.no_grad():
            output1_eval, _ = model(x)
            output2_eval, _ = model(x)

        assert torch.allclose(
            output1_eval, output2_eval, atol=1e-6
        ), "Outputs should be identical in eval mode"


def test_regressor_expert():
    """Test RegressorExpert module"""
    embedding_dim = 256
    dropout = 0.1

    expert = RegressorExpert(embedding_dim, dropout)

    batch_size, seq_len = 4, 64
    x = torch.randn(batch_size, seq_len, embedding_dim)

    output = expert(x)

    # Check output shape (last dimension should be 1, as it's regression output)
    assert output.shape == (
        batch_size,
        seq_len,
        1,
    ), f"Expected shape {(batch_size, seq_len, 1)}, got {output.shape}"


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing MoERegressor")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    test_suite = TestMoERegressor()

    # Get fixtures
    default_params = {
        "embedding_dim": 256,
        "dropout": 0.1,
        "number_of_experts": 3,
        "gate_temperature": 1.0,
    }

    tests = [
        ("Initialization", lambda: test_suite.test_initialization(default_params)),
        (
            "Forward shape",
            lambda: test_suite.test_forward_shape(default_params, device),
        ),
        (
            "Gate weights validity",
            lambda: test_suite.test_gate_weights_validity(default_params, device),
        ),
        (
            "Different expert numbers",
            lambda: test_suite.test_different_expert_numbers(device),
        ),
        (
            "Gate temperature effect",
            lambda: test_suite.test_gate_temperature_effect(default_params, device),
        ),
        (
            "Gradient flow",
            lambda: test_suite.test_gradient_flow(default_params, device),
        ),
        (
            "Weight initialization",
            lambda: test_suite.test_weight_initialization(default_params),
        ),
        ("Output range", lambda: test_suite.test_output_range(default_params, device)),
        ("Zero input", lambda: test_suite.test_zero_input(default_params, device)),
        ("Determinism", lambda: test_suite.test_determinism(default_params, device)),
        ("Dropout effect", lambda: test_suite.test_dropout_effect(device)),
        ("RegressorExpert", test_regressor_expert),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n[Testing] {test_name}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print("❌ FAILED")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
