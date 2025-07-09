#!/usr/bin/env python3
"""
梯度检查使用示例
"""

import torch
import torch.nn as nn

from deepsc.utils.utils import check_grad_flow


def create_simple_model():
    """创建一个简单的测试模型"""
    model = nn.Sequential(
        nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5), nn.Softmax(dim=-1)
    )
    return model


def test_grad_flow():
    """测试梯度检查功能"""
    print("🔍 测试梯度检查功能...")

    # 创建模型和数据
    model = create_simple_model()
    x = torch.randn(32, 10)
    target = torch.randint(0, 5, (32,))

    # 前向传播
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)

    print(f"Loss: {loss.item():.4f}")

    # 执行梯度检查
    grad_stats = check_grad_flow(model, loss, verbose=True)

    print("\n📊 梯度检查结果:")
    print(f"✅ 有效梯度参数: {len(grad_stats['ok'])}")
    print(f"⚠️ 梯度为0的参数: {len(grad_stats['zero'])}")
    print(f"❌ 无梯度的参数: {len(grad_stats['none'])}")

    return grad_stats


if __name__ == "__main__":
    test_grad_flow()
