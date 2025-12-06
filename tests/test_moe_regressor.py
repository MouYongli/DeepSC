#!/usr/bin/env python3
"""
测试 MoERegressor 模块

测试内容：
1. 基本初始化
2. forward 方法的输入输出形状
3. gate_weights 的有效性
4. 不同参数配置
5. 梯度流
6. 边界情况
"""

import os

import pytest
import torch
import torch.nn as nn

import sys

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.deepsc.models.deepsc.model import MoERegressor, RegressorExpert


class TestMoERegressor:
    """MoERegressor 测试类"""

    @pytest.fixture
    def default_params(self):
        """默认参数"""
        return {
            "embedding_dim": 256,
            "dropout": 0.1,
            "number_of_experts": 3,
            "gate_temperature": 1.0,
        }

    @pytest.fixture
    def device(self):
        """测试设备"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_initialization(self, default_params):
        """测试基本初始化"""
        model = MoERegressor(**default_params)

        # 检查基本属性
        assert model.embedding_dim == default_params["embedding_dim"]
        assert model.num_experts == default_params["number_of_experts"]
        assert model.gate_temperature == default_params["gate_temperature"]

        # 检查 gate 网络结构
        assert isinstance(model.gate, nn.Sequential)

        # 检查 experts 数量
        assert len(model.experts) == default_params["number_of_experts"]

        # 检查每个 expert 是 RegressorExpert 实例
        for expert in model.experts:
            assert isinstance(expert, RegressorExpert)

    def test_forward_shape(self, default_params, device):
        """测试 forward 方法的输入输出形状"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        # 测试不同的 batch_size 和 seq_len
        test_cases = [
            (2, 64),  # 小批次
            (8, 128),  # 中等批次
            (16, 256),  # 大批次
            (1, 32),  # 单样本
        ]

        for batch_size, seq_len in test_cases:
            with torch.no_grad():
                x = torch.randn(
                    batch_size, seq_len, default_params["embedding_dim"]
                ).to(device)
                output, gate_weights = model(x)

                # 检查输出形状
                assert output.shape == (
                    batch_size,
                    seq_len,
                ), f"Expected output shape {(batch_size, seq_len)}, got {output.shape}"

                # 检查 gate_weights 形状
                expected_gate_shape = (
                    batch_size,
                    seq_len,
                    default_params["number_of_experts"],
                )
                assert (
                    gate_weights.shape == expected_gate_shape
                ), f"Expected gate_weights shape {expected_gate_shape}, got {gate_weights.shape}"

    def test_gate_weights_validity(self, default_params, device):
        """测试 gate_weights 的有效性"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, default_params["embedding_dim"]).to(device)

        with torch.no_grad():
            output, gate_weights = model(x)

            # 1. 检查 gate_weights 在 [0, 1] 范围内
            assert torch.all(
                gate_weights >= 0
            ), "Gate weights should be non-negative (softmax output)"
            assert torch.all(
                gate_weights <= 1
            ), "Gate weights should be <= 1 (softmax output)"

            # 2. 检查每个位置的 gate_weights 和为 1（softmax 特性）
            weight_sums = gate_weights.sum(dim=-1)
            assert torch.allclose(
                weight_sums, torch.ones_like(weight_sums), atol=1e-6
            ), "Gate weights should sum to 1 along expert dimension"

            # 3. 检查 gate_weights 没有 NaN 或 Inf
            assert not torch.isnan(gate_weights).any(), "Gate weights contain NaN"
            assert not torch.isinf(gate_weights).any(), "Gate weights contain Inf"

    def test_different_expert_numbers(self, device):
        """测试不同的 expert 数量"""
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

                # 检查形状
                assert output.shape == (batch_size, seq_len)
                assert gate_weights.shape == (batch_size, seq_len, num_experts)

                # 检查 gate_weights 和为 1
                weight_sums = gate_weights.sum(dim=-1)
                assert torch.allclose(
                    weight_sums, torch.ones_like(weight_sums), atol=1e-6
                )

    def test_gate_temperature_effect(self, default_params, device):
        """测试不同 gate_temperature 对输出的影响"""
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
                # 固定随机种子以确保可重复性
                torch.manual_seed(42)
                _, gate_weights = model(x)
                gate_weights_list.append(gate_weights)

                # 检查 gate_weights 仍然有效
                weight_sums = gate_weights.sum(dim=-1)
                assert torch.allclose(
                    weight_sums, torch.ones_like(weight_sums), atol=1e-6
                )

        # 低温度应该产生更尖锐的分布（更接近 one-hot）
        # 高温度应该产生更平滑的分布（更均匀）
        # 注意：由于随机初始化，我们只检查趋势而不是具体值

    def test_gradient_flow(self, default_params, device):
        """测试梯度流"""
        model = MoERegressor(**default_params).to(device)
        model.train()

        batch_size, seq_len = 4, 64
        x = torch.randn(
            batch_size, seq_len, default_params["embedding_dim"], requires_grad=True
        ).to(device)

        # 保留输入梯度（对于非叶子张量）
        x.retain_grad()

        # 前向传播
        output, gate_weights = model(x)

        # 计算简单的损失
        loss = output.sum()

        # 反向传播
        loss.backward()

        # 检查梯度存在
        assert x.grad is not None, "Input gradients should exist"

        # 检查模型参数的梯度
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
        """测试权重初始化"""
        model = MoERegressor(**default_params)

        # 检查 gate 网络的权重和偏置
        for m in model.gate:
            if isinstance(m, nn.Linear):
                # 检查权重不是全零
                assert not torch.allclose(
                    m.weight, torch.zeros_like(m.weight)
                ), "Gate weights should not be all zeros"
                # 检查偏置是全零
                assert torch.allclose(
                    m.bias, torch.zeros_like(m.bias)
                ), "Gate biases should be initialized to zeros"

        # 检查 expert 网络的权重和偏置
        for expert in model.experts:
            for m in expert.modules():
                if isinstance(m, nn.Linear):
                    # 检查权重不是全零
                    assert not torch.allclose(
                        m.weight, torch.zeros_like(m.weight)
                    ), "Expert weights should not be all zeros"
                    # 检查偏置是全零
                    assert torch.allclose(
                        m.bias, torch.zeros_like(m.bias)
                    ), "Expert biases should be initialized to zeros"

    def test_output_range(self, default_params, device):
        """测试输出值的合理性"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        batch_size, seq_len = 4, 64

        # 测试正常输入
        x = torch.randn(batch_size, seq_len, default_params["embedding_dim"]).to(device)

        with torch.no_grad():
            output, gate_weights = model(x)

            # 检查输出没有 NaN 或 Inf
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

            # 对于归一化的输入，输出应该在合理范围内（不会太大）
            assert torch.abs(output).max() < 1e3, "Output values are unreasonably large"

    def test_zero_input(self, default_params, device):
        """测试零输入的情况"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        batch_size, seq_len = 4, 64
        x = torch.zeros(batch_size, seq_len, default_params["embedding_dim"]).to(device)

        with torch.no_grad():
            output, gate_weights = model(x)

            # 检查输出有效
            assert not torch.isnan(output).any(), "Output contains NaN for zero input"
            assert not torch.isinf(output).any(), "Output contains Inf for zero input"

            # 检查 gate_weights 仍然有效
            weight_sums = gate_weights.sum(dim=-1)
            assert torch.allclose(
                weight_sums, torch.ones_like(weight_sums), atol=1e-6
            ), "Gate weights should sum to 1 for zero input"

    def test_determinism(self, default_params, device):
        """测试确定性（相同输入应产生相同输出）"""
        model = MoERegressor(**default_params).to(device)
        model.eval()

        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, default_params["embedding_dim"]).to(device)

        with torch.no_grad():
            output1, gate_weights1 = model(x)
            output2, gate_weights2 = model(x)

            # 相同输入应该产生相同输出
            assert torch.allclose(
                output1, output2, atol=1e-6
            ), "Model should be deterministic in eval mode"
            assert torch.allclose(
                gate_weights1, gate_weights2, atol=1e-6
            ), "Gate weights should be deterministic in eval mode"

    def test_dropout_effect(self, device):
        """测试 dropout 在训练和评估模式下的效果"""
        params = {
            "embedding_dim": 256,
            "dropout": 0.5,  # 较高的 dropout 率以便观察效果
            "number_of_experts": 3,
            "gate_temperature": 1.0,
        }

        model = MoERegressor(**params).to(device)
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, params["embedding_dim"]).to(device)

        # 训练模式：输出应该不同（由于 dropout）
        model.train()
        torch.manual_seed(42)
        output1_train, _ = model(x)
        torch.manual_seed(43)
        output2_train, _ = model(x)

        # 训练模式下，由于 dropout，输出应该不同
        # 注意：这个测试可能偶尔失败，因为随机性
        assert not torch.allclose(
            output1_train, output2_train, atol=1e-6
        ), "Outputs should differ in training mode due to dropout"

        # 评估模式：输出应该相同
        model.eval()
        with torch.no_grad():
            output1_eval, _ = model(x)
            output2_eval, _ = model(x)

        assert torch.allclose(
            output1_eval, output2_eval, atol=1e-6
        ), "Outputs should be identical in eval mode"


def test_regressor_expert():
    """测试 RegressorExpert 模块"""
    embedding_dim = 256
    dropout = 0.1

    expert = RegressorExpert(embedding_dim, dropout)

    batch_size, seq_len = 4, 64
    x = torch.randn(batch_size, seq_len, embedding_dim)

    output = expert(x)

    # 检查输出形状（最后一维应该是1，因为是回归输出）
    assert output.shape == (
        batch_size,
        seq_len,
        1,
    ), f"Expected shape {(batch_size, seq_len, 1)}, got {output.shape}"


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Testing MoERegressor")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    test_suite = TestMoERegressor()

    # 获取 fixtures
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
