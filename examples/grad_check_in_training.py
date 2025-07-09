#!/usr/bin/env python3
"""
在训练过程中使用梯度检查的示例
"""

import torch
import torch.nn as nn
import torch.optim as optim

from deepsc.utils.utils import check_grad_flow


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
        self.layer3 = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def train_with_grad_check():
    """在训练过程中使用梯度检查"""
    print("🚀 开始训练并监控梯度...")

    # 创建模型、优化器和数据
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 模拟训练数据
    x = torch.randn(100, 10)
    target = torch.randint(0, 5, (100,))

    print(f"模型参数总数: {sum(p.numel() for p in model.parameters())}")
    print(
        f"可训练参数数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    # 训练循环
    for epoch in range(3):
        print(f"\n📊 Epoch {epoch + 1}")

        for step in range(10):  # 每个epoch 10步
            # 前向传播
            output = model(x)
            loss = criterion(output, target)

            # 梯度检查（每5步检查一次）
            if step % 5 == 0:
                print(f"\n🔍 [梯度检查] Epoch {epoch + 1}, Step {step}")
                grad_stats = check_grad_flow(
                    model, loss, verbose=False, retain_graph=True
                )

                # 分析梯度状态
                total_params = (
                    len(grad_stats["ok"])
                    + len(grad_stats["zero"])
                    + len(grad_stats["none"])
                )
                ok_ratio = (
                    len(grad_stats["ok"]) / total_params if total_params > 0 else 0
                )

                print(
                    f"📈 梯度健康度: {ok_ratio:.2%} ({len(grad_stats['ok'])}/{total_params})"
                )

                # 检查是否有问题
                if len(grad_stats["zero"]) > len(grad_stats["ok"]):
                    print("⚠️ 警告: 梯度消失可能正在发生")
                if len(grad_stats["none"]) > 0:
                    print("❌ 警告: 发现冻结的参数")

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if step % 5 == 0:
                print(f"Loss: {loss.item():.4f}")

    print("\n✅ 训练完成!")


def test_gradient_vanishing():
    """测试梯度消失的情况"""
    print("\n🧪 测试梯度消失情况...")

    # 创建一个容易发生梯度消失的模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.Sigmoid(),  # Sigmoid容易导致梯度消失
        nn.Linear(5, 3),
        nn.Sigmoid(),
        nn.Linear(3, 2),
        nn.Sigmoid(),
        nn.Linear(2, 1),
        nn.Sigmoid(),
    )

    x = torch.randn(10, 10)
    target = torch.randn(10, 1)
    criterion = nn.MSELoss()

    # 前向传播
    output = model(x)
    loss = criterion(output, target)

    print("🔍 检查梯度消失情况...")
    grad_stats = check_grad_flow(model, loss, verbose=True, retain_graph=False)

    # 分析结果
    total_params = (
        len(grad_stats["ok"]) + len(grad_stats["zero"]) + len(grad_stats["none"])
    )
    if len(grad_stats["zero"]) > len(grad_stats["ok"]):
        print("⚠️ 检测到梯度消失问题!")
        print("💡 建议: 使用ReLU激活函数或调整学习率")
    else:
        print("✅ 梯度传导正常")


if __name__ == "__main__":
    # 运行基本训练示例
    train_with_grad_check()

    # 运行梯度消失测试
    test_gradient_vanishing()
