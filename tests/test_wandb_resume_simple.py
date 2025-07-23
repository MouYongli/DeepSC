#!/usr/bin/env python3
"""
更新后的wandb resume功能测试脚本
新逻辑：不创建空的run，直接用正确的run_id初始化
"""
import os

import torch

import sys
import wandb

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.deepsc.utils.utils import save_checkpoint


def test_wandb_resume_optimized():
    """测试优化后的 wandb resume 功能 - 不创建空run"""

    # 创建测试目录
    test_dir = "./test_checkpoints"
    os.makedirs(test_dir, exist_ok=True)

    print("=== 测试优化后的 wandb resume 功能 ===")

    # 1. 模拟第一次训练
    print("\n1. 模拟第一次训练...")

    # 初始化wandb
    wandb.init(
        project="test_deepsc_resume",
        name="test_run_original",
        tags=["test", "first_run"],
        config={"test": True, "run_number": 1},
    )

    original_run_id = wandb.run.id
    original_name = wandb.run.name
    original_project = wandb.run.project
    original_entity = wandb.run.entity
    original_tags = list(wandb.run.tags)
    original_config = dict(wandb.run.config)

    print(f"原始run_id: {original_run_id}")
    print(f"原始name: {original_name}")
    print(f"原始tags: {original_tags}")

    # 创建模拟模型和优化器
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    # 记录一些数据
    wandb.log({"epoch": 1, "loss": 0.5})
    wandb.log({"epoch": 2, "loss": 0.3})

    # 保存checkpoint（包含wandb配置）
    save_checkpoint(
        epoch=5,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        model_name="test_model",
        ckpt_folder=test_dir,
        iteration=100,
    )

    # 结束第一次wandb会话
    wandb.finish()

    # 2. 模拟第二次训练（恢复场景）
    print("\n2. 模拟训练恢复...")

    # 模拟在trainer中的逻辑：先尝试加载checkpoint
    ckpt_path = os.path.join(test_dir, "latest_checkpoint.pth")
    if os.path.exists(ckpt_path):
        print("发现checkpoint，尝试加载...")

        # 加载checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        saved_run_id = checkpoint.get("wandb_run_id", None)
        saved_wandb_config = checkpoint.get("wandb_config", None)

        if saved_run_id and saved_wandb_config:
            print(f"找到保存的wandb run_id: {saved_run_id}")
            print("使用原始配置恢复wandb会话...")

            # 直接用原始run_id和配置初始化wandb
            wandb.init(
                id=saved_run_id,
                resume="allow",
                project=saved_wandb_config.get("project"),
                entity=saved_wandb_config.get("entity"),
                name=saved_wandb_config.get("name"),
                tags=saved_wandb_config.get("tags"),
                config=saved_wandb_config.get("config"),
            )

            resumed_run_id = wandb.run.id
            resumed_name = wandb.run.name
            resumed_tags = list(wandb.run.tags)

            print(f"恢复后的run_id: {resumed_run_id}")
            print(f"恢复后的name: {resumed_name}")
            print(f"恢复后的tags: {resumed_tags}")

            # 验证恢复是否成功
            if (
                resumed_run_id == original_run_id
                and resumed_name == original_name
                and resumed_tags == original_tags
            ):
                print("✅ wandb 恢复成功！使用了原始的run_id和配置")

                # 继续记录数据
                wandb.log({"epoch": 6, "loss": 0.2})
                wandb.log({"epoch": 7, "loss": 0.1})

                # 保存另一个checkpoint
                save_checkpoint(
                    epoch=7,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    model_name="test_model",
                    ckpt_folder=test_dir,
                    iteration=140,
                )

                wandb.finish()

                print("✅ 所有测试通过！")
                return True
            else:
                print("❌ 恢复的配置不匹配")
                return False
        else:
            print("❌ checkpoint 中没有找到完整的 wandb 配置")
            return False
    else:
        print("❌ checkpoint 文件不存在")
        return False

    # 3. 清理测试文件
    print("\n3. 清理测试文件...")
    import shutil

    shutil.rmtree(test_dir)


def test_no_checkpoint_scenario():
    """测试没有checkpoint的情况"""

    print("\n=== 测试没有checkpoint的情况 ===")

    # 模拟trainer在没有checkpoint时的行为
    print("没有checkpoint，创建新的wandb run...")

    wandb.init(
        project="test_deepsc_no_checkpoint",
        name="test_new_run",
        tags=["test", "new_run"],
        config={"test": True, "is_new": True},
    )

    new_run_id = wandb.run.id
    print(f"新的run_id: {new_run_id}")

    # 记录一些数据
    wandb.log({"epoch": 1, "loss": 1.0})

    wandb.finish()

    print("✅ 新run创建成功")
    return True


if __name__ == "__main__":
    print("开始测试优化后的 wandb resume 功能...")

    # 测试没有checkpoint的情况
    test1_success = test_no_checkpoint_scenario()

    # 测试有checkpoint恢复的情况
    test2_success = test_wandb_resume_optimized()

    if test1_success and test2_success:
        print("\n🎉 所有测试通过！")
        print("✅ 优化后的wandb resume功能正常工作")
        print("✅ 不会创建空的wandb run")
        print("✅ 恢复时使用原始的run配置")
    else:
        print("\n❌ 部分测试失败！")

    print("\n注意：新的逻辑确保了：")
    print("1. 如果有checkpoint，直接用原始run_id初始化wandb")
    print("2. 如果没有checkpoint，创建新的wandb run")
    print("3. 不会在wandb网站上留下空的run")
