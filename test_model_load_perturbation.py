import os
import random

import hydra
import torch
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig

from deepsc.utils.utils import setup_logging
from src.deepsc.utils import count_unique_cell_types


def _extract_state_dict(maybe_state):
    """
    兼容多种保存方式：
    - {"model": state_dict, ...}  ← 你现在的保存方式（Fabric）
    - {"state_dict": state_dict, ...}  ← Lightning 常见
    - 直接就是 state_dict
    - 键带 "model." 前缀
    """
    if isinstance(maybe_state, dict):
        if "model" in maybe_state and isinstance(maybe_state["model"], dict):
            sd = maybe_state["model"]
        elif "state_dict" in maybe_state and isinstance(
            maybe_state["state_dict"], dict
        ):
            sd = maybe_state["state_dict"]
        else:
            # 可能就是 state_dict
            sd = maybe_state
    else:
        raise ValueError("Checkpoint 内容不是字典，无法解析 state_dict")

    # 去掉可能存在的前缀 "model." 或 "module."
    need_strip_prefixes = ("model.", "module.")
    if any(any(k.startswith(p) for p in need_strip_prefixes) for k in sd.keys()):
        new_sd = {}
        for k, v in sd.items():
            for p in need_strip_prefixes:
                if k.startswith(p):
                    k = k[len(p) :]
                    break
            new_sd[k] = v
        sd = new_sd
    return sd


def _report_loading_result(load_info):
    missing = list(load_info.missing_keys)
    unexpected = list(load_info.unexpected_keys)
    print(f"[LOAD] missing_keys: {len(missing)} | unexpected_keys: {len(unexpected)}")
    if missing:
        print("  - (前10条) missing:", missing[:10])
    if unexpected:
        print("  - (前10条) unexpected:", unexpected[:10])


def _sample_weight_norms(model, sd, k=5):
    """
    随机抽样 k 个双方都存在的参数，打印加载前后的范数变化。
    范数有变化 => 基本可确认成功写入。
    """
    with torch.no_grad():
        common_keys = [name for name, _ in model.named_parameters() if name in sd]
        if not common_keys:
            print("[LOAD] 没有找到与 checkpoint 对齐的公共参数名，无法做范数对比。")
            return
        sample = random.sample(common_keys, min(k, len(common_keys)))
        print("[LOAD] 抽样参数范数对比（加载前 -> 加载后）：")
        for name in sample:
            p = dict(model.named_parameters())[name]
            before = p.detach().float().norm().item()
            # 暂存当前权重
            old = p.detach().cpu().clone()
            # 用 ckpt 覆盖一次
            p.copy_(sd[name].to(p.device).to(p.dtype))
            after = p.detach().float().norm().item()
            print(f"  - {name}: {before:.6f} -> {after:.6f}")
            # 还原（只用于对比；真正的加载在 load_state_dict 里会再做一次）
            p.copy_(old.to(p.device).to(p.dtype))


@hydra.main(version_base=None, config_path="./configs/finetune", config_name="finetune")
def finetune(cfg: DictConfig):
    cfg.cell_type_count = count_unique_cell_types(cfg.data_path, cfg.obs_celltype_col)
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.num_device,
        num_nodes=cfg.num_nodes,
        strategy=DDPStrategy(
            find_unused_parameters=True, gradient_as_bucket_view=False
        ),
        precision="bf16-mixed",
    )
    fabric.launch()
    setup_logging(rank=fabric.global_rank, log_path="./logs")

    # === 实例化模型 ===
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    # （可选）先不转 float，Fabric/AMP 会管；如果你坚持先转，放在 load 之后影响也不大
    # model = model.float()

    # === BEGIN: 仅做“加载是否成功”的测试 ===
    # 1) 路径
    ckpt_path = "/home/angli/baseline/DeepSC/results/latest_checkpoint.ckpt"
    assert os.path.exists(ckpt_path), f"找不到 ckpt: {ckpt_path}"

    # 2) 只在 rank0 读取到 CPU，减少压力；再广播（可选）
    if fabric.global_rank == 0:
        print(f"[LOAD] 读取 checkpoint: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(raw)
    else:
        raw = None
        state_dict = None

    # 3) 广播到所有进程
    state_dict = fabric.broadcast(state_dict, src=0)

    # 4) 打印抽样对比（可选，但很直观）
    _sample_weight_norms(model, state_dict, k=5)

    # 5) 真正加载（strict=False：允许你新增的 embedding 层留空）
    load_info = model.load_state_dict(state_dict, strict=False)
    _report_loading_result(load_info)

    # 6) 简单的形状校验：把双方都有的键对一个，随便挑一个比较一下 shape
    common = [
        (k, v.shape, dict(model.named_parameters())[k].shape)
        for k, v in state_dict.items()
        if k in dict(model.named_parameters())
    ]
    if common:
        k0, s_ckpt, s_model = common[0]
        print(f"[LOAD] 形状示例: {k0}: ckpt {tuple(s_ckpt)} vs model {tuple(s_model)}")

    # 若需要，此时再转 dtype
    # model = model.float()

    # 7) 到这里基本就能确认是否“加载成功”了；如果只是想验证，直接 return 即可：
    if cfg.get("only_check_loading", False):
        print("[LOAD] 仅检查加载，程序结束。")
        return
    # === END: 仅做“加载是否成功”的测试 ===

    # === 下面才是常规训练流程 ===
    # 最好在 setup 之后再构建优化器，以免 DDP/Fabric 包装参数后出现 param mismatch
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)

    # 真正的业务 Trainer（如果需要）
    # 注意：你原来把 PerturbationPrediction 注释掉了；要用就解开
    # trainer = PerturbationPrediction(cfg, fabric=fabric, model=model)
    # trainer.train()


if __name__ == "__main__":
    finetune()
